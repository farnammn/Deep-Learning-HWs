import json
import operator
import numpy as np
import re

class Data_Loader:

    def __init__(self
                 , base_directory
                 , train_annotation = 'mscoco_train2014_annotations.json'
                 , val_annotation = 'mscoco_val2014_annotations.json'
                 , train_multiple_choice = 'MultipleChoice_mscoco_train2014_questions.json'
                 , val_multiple_choice = 'MultipleChoice_mscoco_val2014_questions.jsonf'
                 , train_open_ended = 'OpenEnded_mscoco_train2014_questions.json'
                 , val_open_ended = 'OpenEnded_mscoco_val2014_questions.json'
                 , num_top = 1000
                 , max_questions_words = 23):

        self.data_dictionary = {}
        self.image_ids = {}
        self.questions = {}
        self.answers_array = {}
        self.mask_array = {}

        self.load_data(base_directory + train_annotation, "train_annotation")
        self.load_data(base_directory + val_annotation, "val_annotation")
        self.load_data(base_directory + train_open_ended, "train_open_ended")
        self.load_data(base_directory + val_open_ended, "val_open_ended")
        self.num_top = num_top
        self.wastes = "[?!,.\[\]\>\<]"
        self.max_question_words = max_questions_words
        self.num_train_questions = len(self.data_dictionary["train_open_ended"]["questions"])
        self.num_test_questions = len(self.data_dictionary["train_open_ended"]["questions"])
        self.find_top_questions("train")

        # self.load_all_questions("train")
        # self.load_all_questions("val")


        print("whole data successfully loaded")

    def load_data(self, file_name, dic_name):
        file = open(file_name)
        d = file.readlines()
        self.data_dictionary[dic_name] = json.loads(d[0])

    def find_top_questions(self, function_type = "train"):
        ans_freq = {}
        for anot_data in self.data_dictionary[function_type + "_annotation"]["annotations"]:
            # print(open)
            for i in range(10):
                answer = anot_data["answers"][i]["answer"]
                if answer in ans_freq:
                    ans_freq[answer] = ans_freq[answer] + 1
                else:
                    ans_freq[answer] = 0

        sorted_ans_freq = sorted(ans_freq.items(), key=operator.itemgetter(1))
        top = sorted_ans_freq[-self.num_top:]
        self.top = [item[0] for item in top]
        del top, sorted_ans_freq, ans_freq


    def to_one_hot(self, answers):
        out = np.zeros((len(answers),self.num_top + 1))
        for i in range(len(answers)):
            out[i, answers[i]] = 1
        return out

    def load_questions(self,batch_start, batch_end, function_type = "train"):
        self.image_ids = []
        self.questions = []
        self.answers_array = []
        self.mask_array = []

        open_data, anot_data  = self.data_dictionary[function_type + "_open_ended"]["questions"],\
                                self.data_dictionary[function_type + "_annotation"]["annotations"]
        for j in range(batch_start, batch_end):
            answers = [
                (self.top.index(anot_data[j]["answers"][i]["answer"]) if anot_data[j]["answers"][i]["answer"] in self.top else self.num_top)
                for i in range(10)
                ]

            self.image_ids.append(int(open_data[j]["image_id"]))

            question = open_data[j]["question"]
            question = re.sub(self.wastes, "", question)
            question = question.split(" ")

            mask = [1] * len(question) + [0] * np.maximum(self.max_question_words - len(question), 0)
            question = question + [""] * np.maximum(self.max_question_words - len(question), 0)

            self.questions.append(question)
            self.mask_array.append(mask)
            self.answers_array.append(self.to_one_hot(answers))
        return np.array(self.questions) , np.array(self.answers_array) \
            , self.image_ids , np.array(self.mask_array)














