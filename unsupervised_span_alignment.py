import argparse
import json
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel

from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor, BidirectionalEndpointSpanExtractor

from sacremoses import MosesTokenizer

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str, required=True, help='file with input data')
	parser.add_argument('--output', type=str, help='file to write outputs')
	parser.add_argument('--source-lang', type=str, required=True, help='language of source text')
	parser.add_argument('--target-lang', type=str, required=True, help='language of target text')
	parser.add_argument('--max-source-span-width', type=int, default=None, help='maximum length of a source text span')
	parser.add_argument('--max-target-span-width', type=int, default=None, help='maximum length of a target text span')
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

def align(source_tokens, target_tokens, tokenizer, model, span_extractor, source_span_boundaries=None, target_span_boundaries=None, source_lang="en", target_lang="fr", max_source_span_width=None, max_target_span_width=None, verbose=False):
	if verbose:
		print('='*20)
		print(f"src: {source_tokens}")
		print(f"tgt: {target_tokens}")
		print('-'*20)

	encoded_source_sentence = tokenizer(source_tokens, padding=True, truncation=True, return_tensors='pt', is_split_into_words=True)
	encoded_target_sentence = tokenizer(target_tokens, padding=True, truncation=True, return_tensors='pt', is_split_into_words=True)

	source_subtoken_ids = encoded_source_sentence['input_ids'][0].tolist()
	target_subtoken_ids = encoded_target_sentence['input_ids'][0].tolist()

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


	if not source_span_boundaries:
		source_span_boundaries = enumerate_spans(source_tokens, max_span_width=max_source_span_width)
	if not target_span_boundaries:
		target_span_boundaries = enumerate_spans(target_tokens, max_span_width=max_target_span_width)

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
		source_subtoken_embeddings = model(**encoded_source_sentence)['last_hidden_state']
		target_subtoken_embeddings = model(**encoded_target_sentence)['last_hidden_state']
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
				table[i][j] = torch.nn.functional.cosine_similarity(source_span_embeddings[i], target_span_embeddings[j], dim=0)

	used_source_spans = []
	used_target_spans = []
	result = []
	for _ in range(table.size):
		best_alignment = np.unravel_index(table.argmax(), table.shape)
		source_span_idx, target_span_idx = best_alignment
		source_span = source_span_boundaries[source_span_idx]
		target_span = target_span_boundaries[target_span_idx]

		if not overlaps(source_span[0], source_span[1], used_source_spans) and not overlaps(target_span[0], target_span[1], used_target_spans):
			source_span_tokens = source_tokens[source_span[0]:source_span[1]+1]
			target_span_tokens = target_tokens[target_span[0]:target_span[1]+1]

			result.append({"source_span": source_span, "target_span": target_span, "source_span_tokens": source_span_tokens, "target_span_tokens": target_span_tokens, "score": table[best_alignment]})

			if verbose:
				print(source_span_tokens, target_span_tokens, source_span, target_span, table[best_alignment])

			used_source_spans.append(source_span)
			used_target_spans.append(target_span)

		table[best_alignment] = float('-inf')  # erase so we can move on to the next best alignment

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

	tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
	model = AutoModel.from_pretrained('xlm-roberta-base')
	span_extractor = mean_pool
	# span_extractor = EndpointSpanExtractor(768)

	# https://aclanthology.org/2020.repl4nlp-1.20/
	# span_extractor = EndpointSpanExtractor(768, combination="x+y,y-x")  # Diff-Sum

	def _tokenize(sentence, lang):
		return MosesTokenizer(lang=lang).tokenize(sentence, escape=False)

	outputs = []
	for item in data:
		if type(item['source']) in [tuple, list]:
			source_sentence = item['source']
		elif type(item['source']) == str:
			source_sentence = _tokenize(item['source'], lang=args.source_lang)
		else:
			raise TypeError(f"cannot handle source text with type: {type(item['source'])}")

		if type(item['target']) in [tuple, list]:
			target_sentence = item['target']
		elif type(item['target']) == str:
			target_sentence = _tokenize(item['target'], lang=args.target_lang)
		else:
			raise TypeError(f"cannot handle target text with type: {type(item['target'])}")

		source_span_boundaries = item.get('source_spans', None)
		target_span_boundaries = item.get('target_spans', None)

		alignment = align(source_sentence, target_sentence, tokenizer, model, span_extractor, source_span_boundaries, target_span_boundaries, max_source_span_width=args.max_source_span_width, max_target_span_width=args.max_target_span_width, verbose=args.verbose)

		outputs.append({"source_tokens": source_sentence, "target_tokens": target_sentence, "alignment": alignment})

	if args.output:
		with open(args.output, "w") as f:
			for output in outputs:
				f.write(json.dumps(output, ensure_ascii=False))
				f.write('\n')

if __name__ == "__main__":
	main()
