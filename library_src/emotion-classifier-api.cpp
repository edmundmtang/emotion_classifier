#include "emotion-classifier-api.h"

Classifier::Classifier(std::string model_path, std::string vocab_path, int max_len, float threshold) {
	LoadModel(model_path);
	LoadVocab(vocab_path);

	MAX_LEN = max_len;
	THRESHOLD = threshold;
}

Classifier::~Classifier() {
	/*I'm not sure what sort of cleanup needs to be performed.*/
}

void Classifier::LoadModel(std::string model_path) {
	try {
		model = torch::jit::load(model_path);
	}
	catch (const c10::Error& e) {
		std::cerr << "Error loading model\n";
	}
	model.eval();
}

void Classifier::LoadVocab(std::string vocab_path) {
	std::fstream newfile;
	newfile.open(vocab_path, std::ios::in);
	if (newfile.good() == false)
		std::cout << "Error loading vocabulary\n";

	std::string line;
	int token_id = 0;
	while (getline(newfile, line)) {
		char* token = strtok(const_cast<char*>(line.c_str()), " ");
		token2id[token] = token_id;
		id2token[token_id] = token;
		token_id++;
	}
	newfile.close();

	/*We might implement a generalized loading of tokens
	from a file here.*/
	pad_token = "[PAD]";
	start_token = "[CLS]";
	end_token = "[SEP]";
	unk_token = "[UNK]";

	pad_token_id = token2id[pad_token];
	start_token_id = token2id[start_token];
	end_token_id = token2id[end_token];
}

std::string Classifier::ProcessText(std::string text) {
	/*Partially perform the preprocessing step and return
	a string demonstrating the input string prior to conversion
	to token ids. For debugging purposes.*/
	std::istringstream input_ss(text);
	std::stringstream ss("", std::ios::app | std::ios::out | std::ios::in);
	ss = SplitOnPunc(input_ss);
	ss = WordPieceTokenize(ss);
	return ss.str();
}

std::pair<torch::Tensor, torch::Tensor> Classifier::PreProcessText(std::istringstream input_ss) {
	/*Convert an input stringstream into an input_ids tensor and
	a masks tensor for use in the pytorch classification model.
	This is assuming that the input is already lowercase, features
	typical whitespacing, and doesn't contain accents nor non-
	English characters.*/
	std::stringstream ss("", std::ios::app | std::ios::out | std::ios::in);
	ss = SplitOnPunc(input_ss);
	ss = WordPieceTokenize(ss);

	std::vector<int> input_ids(MAX_LEN, pad_token_id), masks(MAX_LEN, 0);
	input_ids[0] = start_token_id; masks[0] = 1;
	std::string word;

	int input_id = 1;
	while (getline(ss, word, ' ')) {
		int word_id = token2id[word];
		masks[input_id] = 1;
		input_ids[input_id++] = word_id;
	}

	masks[input_id] = 1;
	input_ids[input_id] = end_token_id;

	auto input_ids_tensor = torch::tensor(input_ids).unsqueeze(0);
	auto masks_tensor = torch::tensor(masks).unsqueeze(0);
	return std::make_pair(input_ids_tensor, masks_tensor);
}

std::stringstream Classifier::SplitOnPunc(std::istringstream& input_ss) {
	/*Split words with punctuation in them into separate
	tokens. Punctuation is treated as its own tokens.*/
	std::string word;
	std::stringstream output_ss("", std::ios::app | std::ios::out | std::ios::in);
	std::regex rgx("[.,\'\"]"); // expecting only basic punctuation
	std::smatch base_match;
	while (getline(input_ss, word, ' ')) {
		bool start_new_word(true);
		if (output_ss.str() == "")
			start_new_word = false;
		for (char& ch : word) {
			std::string ch_str(1, ch);
			if (std::regex_match(ch_str, base_match, rgx)) {
				output_ss << " ";
				start_new_word = true; // there is probably an issue if the first symbol of the entire ss is punctuation
			}
			else {
				if (start_new_word) {
					output_ss << " ";
				}
				start_new_word = false;
			}
			output_ss << ch;
		}
	}
	return output_ss;
}

std::stringstream Classifier::WordPieceTokenize(std::stringstream& input_ss) {
	/*Split words into sub tokens. Some words are treated as
	the compound of multiple tokens.*/
	std::stringstream output_ss("", std::ios::app | std::ios::out | std::ios::in);
	std::string token;
	while (getline(input_ss, token, ' ')) {
		int start(0);
		bool is_bad = false;
		std::stringstream subtoken_ss("", std::ios::app | std::ios::out | std::ios::in);
		while (start < token.length()) {
			int end = token.length();
			std::string cur_substr("");
			while (start < end) {
				std::string substr = token.substr(start, end - start);
				if (start > 0)
					substr = "##" + substr;
				if (auto search = token2id.find(substr); search != token2id.end()) {
					cur_substr = substr;
					break;
				}
				end--;
			}
			if (cur_substr == "") {
				is_bad = true;
				break;
			}
			if (output_ss.str() == "" && subtoken_ss.str() == "") {
				subtoken_ss << cur_substr;
			}
			else {
				subtoken_ss << " " << cur_substr;
			}
			start = end;
		}
		if (is_bad) {
			if (output_ss.str() == "") {
				output_ss << unk_token;
			}
			else {
				output_ss << " " << unk_token;
			}
		}
		else {
			output_ss << subtoken_ss.str();
		}
	}
	return output_ss;
}

int Classifier::ClassifyText(std::string text) {
	/*Take a text string and return an int predicting the
	emotional content of the string. Full pipeline.*/
	std::istringstream input_ss(text);
	torch::Tensor input_ids, masks;

	//std::tie(input_ids, masks) = PreProcessText(input_ss);
	//std::vector<torch::jit::IValue> inputs;
	//inputs.push_back(input_ids);
	//inputs.push_back(masks);

	//torch::IValue result = model.forward(inputs);
	//return PostProcessResult(result);
	return 1;
}

int Classifier::PostProcessResult(torch::IValue result) {
	torch::Tensor result_tensor = result.toTuple()->elements()[0].toTensor();
	return ThresholdSoftmax(result_tensor);
}

int Classifier::ThresholdSoftmax(torch::Tensor result) {
	torch::Tensor softmax_result = torch::softmax(result, 1);
	if (torch::max(softmax_result).item<float>() > THRESHOLD) {
		return torch::argmax(softmax_result).item<int>();
	}
	else {
		return -1;
	}
}