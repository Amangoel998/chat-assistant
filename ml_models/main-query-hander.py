import pickle
import random
from predictionHelpers import classify_local

maps = {}
responses = {}
with open('./contexts.pkl', "rb") as f:
    maps = pickle.Unpickler(f).load()
with open('./responses.pkl', "rb") as f:
    responses = pickle.Unpickler(f).load()
# print(responses)
# print(maps)
predicted_class = classify_local("""Hello

How are you?

""")[0]
# predicted_class = classify_local("how many datastore in datacenter")[0]
# predicted_class = classify_local("What is the host hardware ?")[0]
# predicted_class = classify_local("how many guest os in virtual machine")[0]

question = None
if float(predicted_class[1])<.75:
    response = responses.get('noanswer')
    print('No answer:', response[random.randint(0,len(response)-1)])
elif predicted_class[0][:6]=='common':
    response = responses.get(predicted_class[0])
    print('Common', response[random.randint(0,len(response)-1)])
elif predicted_class[0] in maps.keys():
    question = maps.get(predicted_class[0])
    print('Question', question)
else:
    print('API Call', predicted_class[0])