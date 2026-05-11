import pandas as pd
import json
import pickle

class XESE35M:
    def __init__(self,input_csv_path, kc_route_path, question_path):
        self.input_csv_path = input_csv_path
        self.kc_route_path = kc_route_path
        self.question_path = question_path
        self.df = pd.read_csv(self.input_csv_path)
        
    def return_csv(self,output_csv_path):
        NEED_COLUMNS = ['uid', 'questions', 'responses', 'timestamps']
        self.df = self.df[NEED_COLUMNS]
        for COL in ['questions', 'responses', 'timestamps']:
            self.df[COL] = self.df[COL].astype(str).str.split(',')

        self.df = self.df.explode(['questions', 'responses', 'timestamps'])

        self.df['timestamps'] = pd.to_numeric(self.df['timestamps'], errors='coerce')
        self.df['timestamps'] = pd.to_datetime(self.df['timestamps'], unit='ms')
        self.df['responses'] = pd.to_numeric(self.df['responses'])
        self.df['timestamps'] = pd.to_numeric(self.df['timestamps'])


        self.df = self.df[
            (self.df['questions'] != -1) &
            (self.df['responses'] != -1) &
            (self.df['timestamps'] != -1)
        ]
        
        # bỏ duplicate hoàn toàn
        self.df = self.df.drop_duplicates(
            subset=['uid', 'questions', 'responses', 'timestamps'],
            keep='first'
        )
        
        self.df = self.df.sort_values(by='timestamps').reset_index(drop=True)
        self.df.to_csv(output_csv_path)
    def return_question_dict(self,output_dict_path):
        data_json = json.load(open(self.question_path))
        
        result = {}

        for key,value in data_json.items():
            
            id = int(key)

            content = value.get("content","")
            concepts = value.get("kc_routes", [])

            result[id] = {
            "content": content,
            "kc_routes": concepts
        } 
            
        with open(output_dict_path, "wb") as f:   # wb = write binary
            pickle.dump(result, f)
    

    def return_kc_dict(self,output_dict_path):
        data_json = json.load(open(self.kc_route_path))
        
        result = {}
        for key,value in data_json.items():

            id = int(key)
            result[id] = value

        with open(output_dict_path, "wb") as f:   # wb = write binary
            pickle.dump(result, f)


data = XESE35M(input_csv_path="dataset/XES3G5M/train_valid_sequences.csv", kc_route_path="dataset/XES3G5M/kc_routes_map_en.json", question_path="dataset/XES3G5M/questions_en.json")
data.return_csv("dataset/processed/XES3G5M/processed.csv")
data.return_kc_dict("dataset/processed/XES3G5M/kc_dict.pkl")
data.return_question_dict("dataset/processed/XES3G5M/question_dict.pkl")
