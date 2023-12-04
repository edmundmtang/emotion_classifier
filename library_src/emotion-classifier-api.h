#ifndef EMOTION_CLASSIFIER_API_H
#define EMOTION_CLASSIFIER_API_H

#include <torch/torch.h>
#include <torch/script.h>

#include <string>
#include <cstring>
#include <sstream>
#include <istream>
#include <fstream>
#include <regex>
#include <vector>
#include <iostream>
#include <memory>


class Classifier {
	public:
		Classifier(std::string model_path, std::string vocab_path, int max_len, float threshold);
		~Classifier();

		std::string ProcessText(std::string text);
		int ClassifyText(std::string text);
		
	private:
		void LoadModel(std::string model_path);
		void LoadVocab(std::string vocab_path);
		
		std::pair<torch::Tensor, torch::Tensor> PreProcessText(std::istringstream input_ss);
		std::stringstream SplitOnPunc(std::istringstream& input_ss);
		std::stringstream WordPieceTokenize(std::stringstream& input_ss);

		int PostProcessResult(torch::IValue result);
		int ThresholdSoftmax(torch::Tensor result);

		torch::jit::script::Module model;
		std::map<std::string, int> token2id;
		std::map<int, std::string> id2token;
		int MAX_LEN = 256;
		float THRESHOLD = 0.80;

		std::string pad_token = "[PAD]";
		std::string start_token = "[CLS]";
		std::string end_token = "[SEP]";
		std::string unk_token = "[UNK]";

		int pad_token_id = 0;
		int start_token_id = 101;
		int end_token_id = 102;
};

#endif /*EMOTION_CLASSIFIER_API_H*/