#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <map>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>
#include <vector>
#include <istream>

#include <regex>


// See https://github.com/google-research/bert/blob/master/tokenization.py for pytorch's implementation of tokenization in python

std::stringstream split_on_punc(std::string text, bool log = false) {
    if (log)
        std::cout << "Split on punctuation...\n";
    std::istringstream input_ss(text);
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
                start_new_word = true;
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
    if (log) {
        std::cout << "Output: " << output_ss.str() << "\n";
    }
    return output_ss;
}

std::stringstream wordpiece_tokenize(std::stringstream& input_ss, std::map<std::string, int> token2id, bool log = false) {
    if (log)
        std::cout << "Wordpiece tokenize...\n";
    std::string unk_token = "[UNK]";
    std::stringstream output_ss("", std::ios::app | std::ios::out | std::ios::in);
    std::string token;
    while (getline(input_ss, token, ' ')) {
        int start(0);
        bool is_bad = false;
        std::stringstream subtoken_ss("", std::ios::app | std::ios::out | std::ios::in);
        while (start < token.length()) {
            int end = token.length();
            int n = end - start;
            std::string cur_substr("");
            while (start < end) {
                std::string substr = token.substr(start, end-start);
                if (start > 0)
                    substr = "##" + substr;
                if (auto search = token2id.find(substr); search != token2id.end()) {
                    cur_substr = substr;
                    break;
                }
                end--;
            }
            if (cur_substr == "")
            {
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
    if (log)
        std::cout << "Output: " << output_ss.str() << "\n";
    return output_ss;
}

std::pair<torch::Tensor, torch::Tensor> preprocess(std::string text, std::map<std::string, int> token2id, int max_length, bool is_lower = true, bool log = false) {
    std::string pad_token = "[PAD]", start_token = "[CLS]", end_token = "[SEP]";
    int pad_token_id = token2id[pad_token], start_token_id = token2id[start_token], end_token_id = token2id[end_token];

    std::vector<int> input_ids(max_length, pad_token_id), masks(max_length, 0);
    input_ids[0] = start_token_id; masks[0] = 1;

    // Assuming text is already in unicode and does not require invalid character removal

    if (!is_lower) {
        std::cout << "Warning: we're not using all lowercase?\n";
        // To-do: we'd normally need to convert to lowercase here, but
        // our input from upstream is lowercase so we'll not bother
        // To-do: we'd also need to strip accents, but I'm pretty sure
        // we're not passing words with accents
    }

    // We do not need to implement a whitespace_tokenize because our
    // inputs should already have the appropriate whitespacing

    std::string word;
    std::stringstream ss("", std::ios::app | std::ios::out | std::ios::in);

    ss = split_on_punc(text, log);
    ss = wordpiece_tokenize(ss, token2id, log);

    int input_id = 1;
    while(getline(ss, word, ' ')) {
      int word_id = token2id[word];
      masks[input_id] = 1;
      input_ids[input_id++] = word_id;

      if (log)
        std::cout << word << " : " << word_id << '\n';
    }

    masks[input_id] = 1;
    input_ids[input_id] = end_token_id;

    if (log){
    for (auto i : input_ids)
        std::cout << i << ' ';
    std::cout << '\n';

    for (auto i : masks)
        std::cout << i << ' ';
    std::cout << '\n';
    }

    auto input_ids_tensor = torch::tensor(input_ids).unsqueeze(0);
    auto masks_tensor = torch::tensor(masks).unsqueeze(0).unsqueeze(0);
    return std::make_pair(input_ids_tensor, masks_tensor);
}

struct Model{
    int max_length = 256;
    std::map<std::string, int> token2id;
    std::map<int, std::string> id2token;
    torch::jit::script::Module bert;

    void init_vocab(std::string vocab_path = "bert-based-uncased-vocab.txt"){
        std::tie(token2id, id2token) = get_vocab(vocab_path);
    }

    std::pair<std::map<std::string, int>, std::map<int, std::string>> get_vocab(std::string vocab_path){
    std::map<std::string, int> token2id;
    std::map<int, std::string> id2token;

    std::fstream newfile;

    newfile.open(vocab_path, std::ios::in);

    std::string line;
    int token_id = 0;
    while (getline(newfile, line)) {
        char *token = strtok(const_cast<char*>(line.c_str()), " ");

        token2id[token] = token_id;
        id2token[token_id] = token;
        token_id++;
    }
    newfile.close();

    return std::make_pair(token2id, id2token);
    }
};

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: emotion-classifier <text-to-analyze>\n";
        return -1;
    }

    std::cout << "Start!" << std::endl;
    int seed = 42;
    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);

    auto model = Model();
    model.init_vocab();

    auto token2id = model.token2id;

    torch::Tensor input_ids, masks;

    std::tie(input_ids, masks) = preprocess(argv[1], token2id, model.max_length, true, true);

    std::cout << "Done\n";

    return 0;
}