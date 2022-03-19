import argparse
import json
import numpy as np
import torch

import benepar, spacy
from spacy.tokens import Doc
from benepar import BeneparComponent

from transformers import AutoTokenizer, AutoModel, AutoConfig

from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor, BidirectionalEndpointSpanExtractor

from sacremoses import MosesTokenizer

SPACY_NAMES = {"en": {"spacy_model_name": 'en_core_web_md', "parser_name": 'benepar_en3'},
			   "fr": {"spacy_model_name": 'fr_core_news_md', "parser_name": 'benepar_fr2'},
			  }

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str, required=True, help='file with input data')
	parser.add_argument('--output', type=str, help='file to write outputs')
	parser.add_argument('--source-lang', type=str, required=True, help='language of source text')
	parser.add_argument('--target-lang', type=str, required=True, help='language of target text')
	parser.add_argument('--max-source-span-width', type=int, default=None, help='maximum length of a source text span')
	parser.add_argument('--max-target-span-width', type=int, default=None, help='maximum length of a target text span')
	parser.add_argument('--span-enumeration', type=str, required=True, choices=['full', 'parse'], help='how to enumerate spans when not provided')
	parser.add_argument('--span-extractor', type=str, choices=['mean_pool', 'max_pool', 'mean_max_pool', 'endpoint', 'diffsum', 'coherent'], help='which span extractor to use')
	parser.add_argument('--decoding-method', type=str, required=True, choices=['greedy', 'intersection', 'weighted_intersection'], help='which method to use to decode alignments')
	parser.add_argument('--allow-overlap', action='store_true', help='whether to allow spans in the alignments to overlap')
	parser.add_argument('--pretrained-encoder-name', type=str, default='microsoft/xlm-align-base', help='which pretrained encoder to use')
	parser.add_argument('--model-layer', type=int, required=True, help='Which layer of the model to get representations from (0-indexed, supports negative indexing)')
	parser.add_argument('--couple', action='store_true', help='whether to couple the representations by encoding the sentences together (otherwise sentences are encoded separately)')
	parser.add_argument('--verbose', action='store_true', help='enable verbose logging')

	args = parser.parse_args()

	return args

def enumerate_spans(tokens, offset=0, min_span_width=1, max_span_width=None, filter_function=None):
	max_span_width = max_span_width or len(tokens)
	filter_function = filter_function or (lambda x: True)
	spans = []

	for start_index in range(len(tokens)):
		first_end_index = min(start_index + min_span_width - 1, len(tokens))
		last_end_index = min(start_index + max_span_width, len(tokens))
		for end_index in range(first_end_index, last_end_index):
			start = offset + start_index
			end = offset + end_index
			# add 1 to end index because span indices are inclusive.
			if filter_function(tokens[slice(start_index, end_index + 1)]):
				spans.append((start, end))

	return spans

def overlaps(i, j, spans):
	minval = min([i] + [s[0] for s in spans])  # smallest start index among spans
	maxval = max([j] + [s[1] for s in spans])  # largest end index among spans
	used_positions = [0]*(maxval - minval + 1)
	for (span_start, span_end) in spans:
		assert used_positions[span_start-minval: span_end-minval+1] == [0]*(span_end-span_start+1)  # check that no span has covered these positions, which should be the case because spans aren't allowed to overlap
		used_positions[span_start-minval: span_end-minval+1] = [1]*(span_end-span_start+1)  # (inclusive) interval has now been covered by a span

	return sum(used_positions[i-minval: j-minval+1]) > 0

def span_mask(subtoken_embeddings, subtoken_span_boundaries):
	subtoken_positions = torch.arange(0,subtoken_embeddings.shape[1]).unsqueeze(-1).repeat(1,subtoken_span_boundaries.shape[0])
	mask = (subtoken_positions.ge(subtoken_span_boundaries[:,0]) * subtoken_positions.le(subtoken_span_boundaries[:,1])).permute(1,0)
	return mask

def mean_pool(subtoken_embeddings, subtoken_span_boundaries):
	mask = span_mask(subtoken_embeddings, subtoken_span_boundaries)

	# Adapted from AllenNLP's `masked_mean()` function
	# (https://github.com/allenai/allennlp/blob/main/allennlp/nn/util.py)
	replaced_vector = subtoken_embeddings.masked_fill(~mask.unsqueeze(-1), 0.0)
	value_sum = torch.sum(replaced_vector, dim=1)
	value_count = torch.sum(mask.unsqueeze(-1), dim=1)
	mean = value_sum / value_count.float().clamp(min=1e-13)

	return mean

def max_pool(subtoken_embeddings, subtoken_span_boundaries):
	mask = span_mask(subtoken_embeddings, subtoken_span_boundaries)

	# Adapted from AllenNLP's `masked_max()` function
	# (https://github.com/allenai/allennlp/blob/main/allennlp/nn/util.py)
	replaced_vector = subtoken_embeddings.masked_fill(~mask.unsqueeze(-1), torch.finfo(subtoken_embeddings.dtype).min)
	max_value, _ = replaced_vector.max(dim=1)

	return max_value

def mean_max_pool(subtoken_embeddings, subtoken_span_boundaries):
	return torch.cat([mean_pool(subtoken_embeddings, subtoken_span_boundaries), max_pool(subtoken_embeddings, subtoken_span_boundaries)], dim=1)

def coherent(subtoken_embeddings, subtoken_span_boundaries):
	embedding_size = subtoken_embeddings.shape[-1]
	endpoint_representation = EndpointSpanExtractor(embedding_size)._embed_spans(subtoken_embeddings, torch.as_tensor(subtoken_span_boundaries).unsqueeze(0))[0]
	# Choice of `a` and `b` dimensionalities follows from the proportions described in footnote 2 of https://aclanthology.org/2020.repl4nlp-1.20/
	a = int((480/1024)*embedding_size)
	b = int((embedding_size - (2*a)) / 2)
	left_endpoint = endpoint_representation[:, :embedding_size]
	right_endpoint = endpoint_representation[:, embedding_size:]
	left_endpoint_pieces = left_endpoint[:, :a], left_endpoint[:, a:a*2], left_endpoint[:, a*2:a*2 + b], left_endpoint[:, a*2 + b:]
	right_endpoint_pieces = right_endpoint[:, :a], right_endpoint[:, a:a*2], right_endpoint[:, a*2:a*2 + b], right_endpoint[:, a*2 + b:]
	coherence_term = torch.sum(left_endpoint_pieces[2]*right_endpoint_pieces[3], dim=1).unsqueeze(-1)
	coherent_representation = torch.cat([left_endpoint_pieces[0], right_endpoint_pieces[1], coherence_term], dim=-1)

	return coherent_representation

def _custom_sentencizer(doc):
	for i, token in enumerate(doc):
		if i == 0:
			doc[i].is_sent_start = True
		else:
			doc[i].is_sent_start = False
	return doc

def align(source_tokens, target_tokens, tokenizer, model, model_layer, span_extractor, decoding_method, source_span_boundaries=None, target_span_boundaries=None, source_lang="en", target_lang="fr", max_source_span_width=None, max_target_span_width=None, allow_overlap=False, couple=False, verbose=False):
	if verbose:
		print('='*20)
		print(f"src: {source_tokens}")
		print(f"tgt: {target_tokens}")
		print('-'*20)

	if couple:
		encoded_sentence_pair = tokenizer(source_tokens + [tokenizer.sep_token, tokenizer.cls_token] + target_tokens, padding=True, truncation=True, return_tensors='pt', is_split_into_words=True)
	else:
		encoded_source_sentence = tokenizer(source_tokens, padding=True, truncation=True, return_tensors='pt', is_split_into_words=True)
		encoded_target_sentence = tokenizer(target_tokens, padding=True, truncation=True, return_tensors='pt', is_split_into_words=True)

	# source_subtoken_ids = encoded_source_sentence['input_ids'][0].tolist()
	# target_subtoken_ids = encoded_target_sentence['input_ids'][0].tolist()

	source_token2subtoken = []
	target_token2subtoken = []

	source_subtokens = [tokenizer.tokenize(token) for token in source_tokens]
	target_subtokens = [tokenizer.tokenize(token) for token in target_tokens]

	assert len(source_tokens) == len(source_subtokens)
	assert len(target_tokens) == len(target_subtokens)

	count = 1  # Account for [CLS] token at beginning of sequence
	for subtokens in source_subtokens:
		source_token2subtoken.append([])
		for subtoken in subtokens:
			source_token2subtoken[-1].append(count)
			count += 1

	count = 1  # Account for [CLS] token at beginning of sequence
	for subtokens in target_subtokens:
		target_token2subtoken.append([])
		for subtoken in subtokens:
			target_token2subtoken[-1].append(count)
			count += 1


	# if not source_span_boundaries:
	# 	source_span_boundaries = enumerate_spans(source_tokens, max_span_width=max_source_span_width)
	# if not target_span_boundaries:
	# 	target_span_boundaries = enumerate_spans(target_tokens, max_span_width=max_target_span_width)

	source_subtoken_span_boundaries = []
	for span_boundary in source_span_boundaries:
		span_boundary_start, span_boundary_end = span_boundary
		adjusted_start = source_token2subtoken[span_boundary_start][0]
		adjusted_end = source_token2subtoken[span_boundary_end][-1]
		source_subtoken_span_boundaries.append((adjusted_start, adjusted_end))

	target_subtoken_span_boundaries = []
	for span_boundary in target_span_boundaries:
		span_boundary_start, span_boundary_end = span_boundary
		adjusted_start = target_token2subtoken[span_boundary_start][0]
		adjusted_end = target_token2subtoken[span_boundary_end][-1]
		target_subtoken_span_boundaries.append((adjusted_start, adjusted_end))

	assert len(source_span_boundaries) == len(source_subtoken_span_boundaries)
	assert len(target_span_boundaries) == len(target_subtoken_span_boundaries)

	num_source_spans = len(source_subtoken_span_boundaries)
	num_target_spans = len(target_subtoken_span_boundaries)
	table = np.zeros((num_source_spans, num_target_spans))

	with torch.no_grad():
		if couple:
			# encode source and target in the same sequence to allow attention flow between the sequences
			num_source_subtokens = len([st for t in source_subtokens for st in t])
			num_target_subtokens = len([st for t in target_subtokens for st in t])
			coupled_subtoken_embeddings = model(**encoded_sentence_pair)['hidden_states'][model_layer]
			assert coupled_subtoken_embeddings.shape[1] == num_source_subtokens + num_target_subtokens + 4  # account for [CLS] and [SEP] tokens surrounding the source and target sequences
			source_subtoken_embeddings = coupled_subtoken_embeddings[:,:num_source_subtokens+2,:]  # account for [CLS] and [SEP] tokens surrounding the source sequence
			target_subtoken_embeddings = coupled_subtoken_embeddings[:,num_source_subtokens+2:,:]
		else:
			source_subtoken_embeddings = model(**encoded_source_sentence)['hidden_states'][model_layer]
			target_subtoken_embeddings = model(**encoded_target_sentence)['hidden_states'][model_layer]

		if type(span_extractor) == SelfAttentiveSpanExtractor:
			source_span_embeddings = span_extractor._embed_spans(source_subtoken_embeddings, torch.tensor(source_subtoken_span_boundaries))[0]
			target_span_embeddings = span_extractor._embed_spans(target_subtoken_embeddings, torch.tensor(target_subtoken_span_boundaries))[0]
		elif type(span_extractor) in [EndpointSpanExtractor, BidirectionalEndpointSpanExtractor]:
			source_span_embeddings = span_extractor._embed_spans(source_subtoken_embeddings, torch.tensor(source_subtoken_span_boundaries).unsqueeze(0))[0]
			target_span_embeddings = span_extractor._embed_spans(target_subtoken_embeddings, torch.tensor(target_subtoken_span_boundaries).unsqueeze(0))[0]
		else:
			source_span_embeddings = span_extractor(source_subtoken_embeddings, torch.tensor(source_subtoken_span_boundaries))
			target_span_embeddings = span_extractor(target_subtoken_embeddings, torch.tensor(target_subtoken_span_boundaries))

		for i in range(num_source_spans):
			for j in range(num_target_spans):
				# TODO: factor into the table scores the span lengths or the difference in span lengths?
				# TODO: do some sort of normalization, like compute F1 as in Eq. 1 of https://arxiv.org/pdf/1910.07475.pdf
				table[i][j] = torch.nn.functional.cosine_similarity(source_span_embeddings[i], target_span_embeddings[j], dim=0)

	# TODO: use some sort of convolution like in https://aclanthology.org/D19-1084/ to encourage locally consistent/monotonic alignments?
	# TODO: use entropy-based null alignment like in https://aclanthology.org/2020.findings-emnlp.147.pdf

	# Decode alignments from table
	if decoding_method == "greedy":
		# This algorithm is similar to the one used in 10.1109/ICMI.2002.1167002 (Huang and Vogel, 2002)
		used_source_spans = []
		used_target_spans = []
		result = []

		# Decode in descending order of score
		for _ in range(table.size):
			best_alignment = np.unravel_index(table.argmax(), table.shape)
			source_span_idx, target_span_idx = best_alignment
			source_span = source_span_boundaries[source_span_idx]
			target_span = target_span_boundaries[target_span_idx]

			if allow_overlap or (not overlaps(source_span[0], source_span[1], used_source_spans) and not overlaps(target_span[0], target_span[1], used_target_spans)):
				source_span_tokens = source_tokens[source_span[0]:source_span[1]+1]
				target_span_tokens = target_tokens[target_span[0]:target_span[1]+1]

				result.append({"source_span": source_span, "target_span": target_span, "source_span_tokens": source_span_tokens, "target_span_tokens": target_span_tokens, "score": table[best_alignment]})

				if verbose:
					print(source_span_tokens, target_span_tokens, source_span, target_span, table[best_alignment])

				used_source_spans.append(source_span)
				used_target_spans.append(target_span)

			table[best_alignment] = float('-inf')  # erase so we can move on to the next best alignment
	elif decoding_method == "intersection":
		used_source_spans = []
		used_target_spans = []
		result = []

		best_source_span_for_target_span = np.argmax(table, axis=0)
		best_target_span_for_source_span = np.argmax(table, axis=1)

		# Convert indices into one-hot vectors
		source_to_target_selection = np.zeros(table.shape)
		source_to_target_selection[np.arange(best_target_span_for_source_span.size), best_target_span_for_source_span] = 1

		target_to_source_selection = np.zeros(table.shape)
		target_to_source_selection[best_source_span_for_target_span, np.arange(best_source_span_for_target_span.size)] = 1

		# Choose alignments that are mutually preferred in both directions (source -> target, target -> source)
		# This is a high precision, low recall approach that incorporates symmetrization into the decoding procedure
		alignments = np.argwhere(source_to_target_selection * target_to_source_selection).tolist()
		for alignment in alignments:
			alignment = tuple(alignment)  # cast to tuple so that indexing into the table works properly
			source_span_idx, target_span_idx = alignment
			source_span = source_span_boundaries[source_span_idx]
			target_span = target_span_boundaries[target_span_idx]

			if allow_overlap or (not overlaps(source_span[0], source_span[1], used_source_spans) and not overlaps(target_span[0], target_span[1], used_target_spans)):
				source_span_tokens = source_tokens[source_span[0]:source_span[1]+1]
				target_span_tokens = target_tokens[target_span[0]:target_span[1]+1]

				result.append({"source_span": source_span, "target_span": target_span, "source_span_tokens": source_span_tokens, "target_span_tokens": target_span_tokens, "score": table[alignment]})

				if verbose:
					print(source_span_tokens, target_span_tokens, source_span, target_span, table[alignment])

				used_source_spans.append(source_span)
				used_target_spans.append(target_span)
	elif decoding_method == "weighted_intersection":
		used_source_spans = []
		used_target_spans = []
		result = []

		best_source_span_for_target_span = np.argmax(table, axis=0)
		best_target_span_for_source_span = np.argmax(table, axis=1)

		source_to_target_selection = np.zeros(table.shape)
		source_to_target_selection[np.arange(best_target_span_for_source_span.size), best_target_span_for_source_span] = 1

		target_to_source_selection = np.zeros(table.shape)
		target_to_source_selection[best_source_span_for_target_span, np.arange(best_source_span_for_target_span.size)] = 1

		# Mask out discarded alignments
		table *= (source_to_target_selection * target_to_source_selection)

		# The most amount of alignments to decode
		# If we use the greedy decoding method here without a way to know when to stop, then we might end up decoding alignments we discarded previously
		max_alignments = len(np.argwhere(table).tolist())

		# Decode in descending order of score
		for _ in range(max_alignments):
			best_alignment = np.unravel_index(table.argmax(), table.shape)
			source_span_idx, target_span_idx = best_alignment
			source_span = source_span_boundaries[source_span_idx]
			target_span = target_span_boundaries[target_span_idx]

			if allow_overlap or (not overlaps(source_span[0], source_span[1], used_source_spans) and not overlaps(target_span[0], target_span[1], used_target_spans)):
				source_span_tokens = source_tokens[source_span[0]:source_span[1]+1]
				target_span_tokens = target_tokens[target_span[0]:target_span[1]+1]

				result.append({"source_span": source_span, "target_span": target_span, "source_span_tokens": source_span_tokens, "target_span_tokens": target_span_tokens, "score": table[best_alignment]})

				if verbose:
					print(source_span_tokens, target_span_tokens, source_span, target_span, table[best_alignment])

				used_source_spans.append(source_span)
				used_target_spans.append(target_span)

			table[best_alignment] = float('-inf')  # erase so we can move on to the next best alignment
	elif decoding_method == "itermax":
		# TODO: implement itermax algorithm from SimAlign: https://aclanthology.org/2020.findings-emnlp.147/
		raise NotImplementedError
	elif decoding_method == "optimal_transport":
		# TODO: implement optimal transport as in Section 3.1 of https://arxiv.org/pdf/2106.06381.pdf
		raise NotImplementedError
	else:
		raise ValueError(f"unrecognized decoding method: {decoding_method}")

	if verbose:
		print('='*20)
		print()

	return result

def main():
	args = parse_args()

	data = []
	with open(args.input, "r") as f:
		for line in f:
			data.append(json.loads(line))

	tokenizer = AutoTokenizer.from_pretrained(args.pretrained_encoder_name)
	model = AutoModel.from_pretrained(args.pretrained_encoder_name, output_hidden_states=True)
	embedding_size = AutoConfig.from_pretrained(args.pretrained_encoder_name).hidden_size

	if args.span_extractor == "mean_pool":
		span_extractor = mean_pool
	elif args.span_extractor == "max_pool":
		span_extractor = max_pool
	elif args.span_extractor == "mean_max_pool":
		span_extractor = mean_max_pool
	elif args.span_extractor == "endpoint":
		span_extractor = EndpointSpanExtractor(embedding_size)
	elif args.span_extractor == "diffsum":
		# https://aclanthology.org/2020.repl4nlp-1.20/
		span_extractor = EndpointSpanExtractor(embedding_size, combination="x+y,y-x")
	elif args.span_extractor == "coherent":
		# https://aclanthology.org/P19-1436/
		span_extractor = coherent
	else:
		raise ValueError(f"unrecognized span extractor: {args.span_extractor}")

	def _get_tokens_and_spans(data, lang, side, spacy_parsers=None):
		tokens = None
		spans = None
		if side == "source":
			sentence = data['source']
			max_span_width = args.max_source_span_width
			given_spans = data.get('source_spans', None)
		elif side == "target":
			sentence = data['target']
			max_span_width = args.max_target_span_width
			given_spans = data.get('target_spans', None)
		else:
			raise ValueError(f"unrecognized side: {side}")

		if type(sentence) in [tuple, list]:
			tokens = sentence
			pretokenized = True
		elif type(sentence) == str:
			pretokenized = False
		else:
			raise TypeError(f"cannot handle text with type: {type(sentence)}")

		if given_spans != None:
			if not pretokenized:
				raise ValueError(f"spans are provided but the text is not pretokenized")

			assert tokens is not None

			return tokens, given_spans

		if not given_spans:
			if args.span_enumeration == "parse":
				if lang not in spacy_parsers.keys():
					raise NotImplementedError(f"Parsing is not supported for {lang}")

				if pretokenized:
					nlp = spacy_parsers[lang]['pretokenized']['spacy_model']
					ConstituencyParser = spacy_parsers[lang]['pretokenized']['parser']
					doc = Doc(nlp.vocab, words=tokens, spaces=[True]*(len(tokens)-1) + [False])
					doc = _custom_sentencizer(doc)
					doc = ConstituencyParser(doc)

				else:
					nlp = spacy_parsers[lang]['untokenized']['spacy_model']
					doc = nlp(sentence)

				if lang == 'en':
					POS_TAGS = ['NP', 'VP']
				elif lang == 'fr':
					POS_TAGS = ['NP', 'VN']
				else:
					raise NotImplementedError(f"Allowable POS tags not specified for {lang}")

				sent = list(doc.sents)[0]
				spans = [(s.start, s.end-1) for s in sent._.constituents for pos in POS_TAGS if pos in s._.labels]
				spans = [span for span in spans if (span[1] - span[0]) <= max_span_width]

				return [str(token) for token in sent], spans
			elif args.span_enumeration == "full":
				if not pretokenized:
					tokens = MosesTokenizer(lang=lang).tokenize(sentence, escape=False)

				spans = enumerate_spans(tokens, max_span_width=max_span_width)
				return tokens, spans

	# Instantiate parsers upfront so we don't have to create a new parser for every example
	SPACY_PARSERS = {}
	if args.span_enumeration == "parse":
		def _instantiate_parser(spacy_model_name, parser_model_name):
			spacy_model = spacy.load(spacy_model_name, disable=['tagger', 'parser', 'ner'])
			parser = BeneparComponent(parser_model_name)

			untok_spacy_model = spacy.load(spacy_model_name, disable=['ner'])
			untok_spacy_model.add_pipe("benepar", config={"model": parser_model_name})

			return {'pretokenized': {"spacy_model": spacy_model, "parser": parser},
					'untokenized': {"spacy_model": untok_spacy_model}}

		for lang in [args.source_lang, args.target_lang]:
			SPACY_PARSERS[lang] = _instantiate_parser(SPACY_NAMES[lang]['spacy_model_name'], SPACY_NAMES[lang]['parser_name'])

	outputs = []
	for item in data:
		source_sentence, source_span_boundaries = _get_tokens_and_spans(item, lang=args.source_lang, side="source", spacy_parsers=SPACY_PARSERS)
		target_sentence, target_span_boundaries = _get_tokens_and_spans(item, lang=args.target_lang, side="target", spacy_parsers=SPACY_PARSERS)

		alignment = align(source_sentence, target_sentence, tokenizer, model, args.model_layer, span_extractor, args.decoding_method, source_span_boundaries, target_span_boundaries, max_source_span_width=args.max_source_span_width, max_target_span_width=args.max_target_span_width, allow_overlap=args.allow_overlap, couple=args.couple, verbose=args.verbose)

		outputs.append({"source_tokens": source_sentence, "target_tokens": target_sentence, "alignment": alignment})

	if args.output:
		with open(args.output, "w") as f:
			for output in outputs:
				f.write(json.dumps(output, ensure_ascii=False))
				f.write('\n')

if __name__ == "__main__":
	main()
