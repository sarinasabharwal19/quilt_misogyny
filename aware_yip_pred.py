#Importing the sequence classification base model and the base tokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
# import sys 
import numpy as np 
from scipy.special import softmax # softmax for the normalized exponential function to normalize the output

class TwitterJoy:
    def __init__(self) -> None:
        # self.base_model = "./twitterismodel"
        self.base_model = "./model"
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.base_model)
        # print(sys.getsizeof(self.model))
    
    def getJoyCode(self,input_text:str):
        print("Getting AWARE code")
        joy_dict = []
        try:
            encoded_input = self.tokenizer(input_text,return_tensors="pt")
        # print(encoded_input)
            output = self.model(**encoded_input)
        # print(output[0][0])
        # print("lenght",len(output[0]))
            scores = output[0][0].detach().numpy()
        #print(scores)
            scores = softmax(scores)
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
        
            labels =  ['Objectification and Belittling','Stereotyping','Flipping the Narrative',
                   'Rape Myths and Victim-Blaming','Violence against women (threats and imagery)']
            
            for i in range(scores.shape[0]):
                lbl = labels[ranking[i]]
                scr = scores[ranking[i]]
                scr = round(scr,2)
                joy_dict.append({"class":lbl,"score":scr})
            return joy_dict
        except:
            pass

if __name__ == '__main__':
    import pandas as pd
    import re
    tj = TwitterJoy()
    data = pd.read_excel("60747bb127a3161b1ada2a36-518_output.xlsx")
    data = data.drop_duplicates(subset=['text'])
    
    for i,v in data.iterrows():
        x = data['text'][i]
        x = str(x)
        x = re.sub('@[^\s]+','',x)
        data['text'][i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", x).split())
    #print(data)
    data["roberta predictions"] = data["text"].apply(lambda x: tj.getJoyCode(x))
    data.to_csv("/content/misogyny_predictions.csv",index=False)
