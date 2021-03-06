from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras

import os
import numpy as np
import json

people_count_in_picture = 20

actions = { 'no_action':0,'standing':1,'falling':2,'moving':3,'setting':4,'digging':5,
            'blocking':6,'spiking':7,'jumping':8,'waiting':9
          }
actions_count = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
actions_one_hot = [i for i in range(len(actions))]
actions_one_hot = to_categorical(actions_one_hot).astype(int)
# print(actions_one_hot)
type_of_actions = len(actions) # one with no action

frames_in_action_series = 41 # how many frame in action series
keypoint_per_person = 75

def load_train_X(dataset_path_x,dataset_path_y):
    missing_label = 0
    dataset_x = []
    dataset_y = []
    for no in os.listdir(dataset_path_x):    # dataset/run
        train_y = load_train_Y(os.path.join(dataset_path_y,no))
        print('loading dataset',no)
        for action_series in os.listdir(os.path.join(dataset_path_x,no)): # dataset/run/1
            single_x =np.array([],dtype=float)
            for action_photos in os.listdir(os.path.join(dataset_path_x,no,action_series)):

                with open(os.path.join(dataset_path_x,no,action_series,action_photos),'r') as f:
                    op = json.load(f)
                    op = json.loads(op)

                people_count = len(op)
                for person_index in range(people_count):
                    for keypoint in op[person_index]:   #25
                        single_x = np.append(single_x,keypoint)    #3
                
                if people_count<people_count_in_picture:
                    single_x = np.append(single_x,np.zeros((people_count_in_picture-people_count)*keypoint_per_person)) 

            try:
                dataset_y.append(train_y[action_series])

                peoples = len(single_x)//(people_count_in_picture*keypoint_per_person)
                if peoples<frames_in_action_series:     # fill in missing frames
                    single_x = np.append(single_x,np.zeros((frames_in_action_series-peoples)*(people_count_in_picture*keypoint_per_person)))
                single_x = single_x.reshape(frames_in_action_series,people_count_in_picture*keypoint_per_person)
                # print(single_x)
                dataset_x.append(single_x)

            except KeyError:
                missing_label += 1

    print('----------------------------------')
    print('load finish','total count :',len(dataset_x),len(dataset_y))    
    print('missing labels :',missing_label)
    print('actions count :',actions_count)

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)
    return (dataset_x,dataset_y,actions_count)

def load_train_Y(dataset_path_y):
    for action_series in os.listdir(dataset_path_y): # dataset/run/1
        if os.path.isfile(os.path.join(dataset_path_y,action_series)):
            dict_y = {}
            with open(os.path.join(dataset_path_y,action_series),'r') as f:
                while True:
                    line = f.readline()
                    if not line:    #EOF
                        break
                    line = line.split('\n')[0].split(' ')
                    persons = (len(line)-2)//5

                    y = np.array([])
                    for i in range(persons):
                        try:
                            y = np.append(y,actions_one_hot[actions[line[2+i*5+4]]])
                            actions_count[actions[line[2+i*5+4]]]+=1
                            # print(actions_one_hot[actions[line[2+i*5+4]]])
                        except KeyError:
                            print(line[2+i*5+4],'not found!')
                            exit()

                    if len(y)//type_of_actions<people_count_in_picture:
                        for i in range(people_count_in_picture-len(y)//type_of_actions):
                            y = np.append(y,actions_one_hot[0])
                            # y = np.pad(y,(0,people_count_in_picture-len(y)//type_of_actions),'constant',constant_values=(0,actions_one_hot[0]))   # fill 0
                            actions_count[0]+=1

                    dict_y[line[0][:-4]] = y    # remove '.jpg'
    return dict_y


def load_dataset(dataset_path_x,dataset_path_y):
    (train_X,train_Y,actions_count) = load_train_X(dataset_path_x,dataset_path_y)
    (train_X,train_Y) = balance_dataset(train_X,train_Y,actions_count)
    # train_X = train_X.astype('float32')
    return (train_X,train_Y)

def balance_dataset(dataset_X,dataset_Y,actions_count):
    # get the minumum amount of less happened action
    min_action_count = actions_count[0]
    for i in range(type_of_actions):
        min_action_count = actions_count[i] if actions_count[i]<min_action_count else min_action_count

    actions_credit = {}
    for i in range(type_of_actions):
        actions_credit[i] = min_action_count
    
    # check if the amount of action exceed the min_action_count
    print(dataset_Y.shape)
    i=0
    while i<len(dataset_Y):
        save = False
        for x in range(people_count_in_picture):
            label = list(dataset_Y[i][type_of_actions*x:type_of_actions*(x+1)]).index(1)
            if(actions_credit[label]>0):
                # not exceed, save
                save = True
                actions_credit[label] -= 1

        if not save:
            # exceed, delete
            dataset_X = np.delete(dataset_X,i,0)
            dataset_Y = np.delete(dataset_Y,i,0)
            i-=1    # remove one element, index move forward
        
        i+=1

    print('----------------------------------')
    print('balance finish','total count :',len(dataset_X),len(dataset_Y))  

    return dataset_X, dataset_Y

def setup_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(41,people_count_in_picture*keypoint_per_person))) #frame ??????node???
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu')) # cell state
    #  fully connection ????????????
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(type_of_actions*people_count_in_picture, activation='sigmoid'))

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    model.summary()
    keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    return model

def train_model(model,train_X,train_Y):
    # print(train_X.shape)
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    model.fit(train_X, train_Y, epochs = 50, batch_size = 32, callbacks=[tb_callback])

def save_model(model,model_path):
    model.save(model_path)

def load_model(model_path):
    return keras.models.load_model(model_path)

def evaluate(model, text_x, test_Y):
    print('val:')
    model.evaluate(x=text_x, y=test_Y, batch_size=32)

def predict(model, test_X, test_Y):
    res = model.predict(test_X)
    print('predict finish!')

    for i in range(5):
        print('result',i)
        print(test_X[i])
        print(res[i])
        print(test_Y[i])

if __name__ == '__main__':
    dataset_path_x = 'F:\\Github\\LSTM-action-detection\\dataset_filiter'
    dataset_path_y = 'F:\\Github\\LSTM-action-detection\\dataset'

    (X,Y) = load_dataset(dataset_path_x,dataset_path_y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    model = setup_model()

    model_path = 'F:\\Github\\LSTM-action-detection\\model\\train.h5'

    train_model(model,X_train,Y_train)
    save_model(model,model_path)

    # model = load_model(model_path)

    # predict(model, X_test, Y_test)
    evaluate(model,X_test,Y_test)