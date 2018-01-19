import tensorflow as tf
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel Bühler for this suggestion
from PIL import Image
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
tf.reset_default_graph()
TRAIN_DIR = '.\\'
IMG_SIZE = 220
LR = 1e-3
def label_img(img,class_name):
    # conversion to one-hot array
    if (class_name == 'potato'):
        if img == 'early_blight': return [1,0,0]

        elif img == 'healthy': return [0,1,0]

        elif img == 'late_blight': return [0,0,1]

    elif (class_name == 'grape'):
        if img == 'blackrot': return [1,0,0,0]

        elif img == 'esca': return [0,1,0,0]

        elif img == 'healthy': return [0,0,1,0]

        elif img == 'leafblight': return [0,0,0,1]

    elif (class_name == 'cucumber'):
        if img == 'healthy':
            return [1, 0]
        elif img == 'downy':
            return [0, 1]

def create_train_data(train_dir,class_name):
    training_data = []
    subfolder_names = []
    leaf_root = os.path.join(TRAIN_DIR, train_dir)
    directories = []
    for filename in os.listdir(leaf_root):
        path = os.path.join(leaf_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            subfolder_names.append(filename)

    i = 0
    for directory in directories:
         for img in tqdm(os.listdir(directory)):
            path = os.path.join(directory, img)
            if (img != 'Thumbs.db'):
                image = Image.open(path)
                image = image.resize((IMG_SIZE, IMG_SIZE))
                label = label_img(subfolder_names[i],class_name)
                training_data.append([np.array(image), np.array(label)])
         i = i + 1
    shuffle(training_data)
    if(class_name=='cucumber'):
        np.save('train_data_cucumber.npy', training_data)
    elif (class_name == 'grape'):
        np.save('train_data_grape.npy', training_data)
    elif (class_name == 'potato'):
        np.save('train_data_potato.npy', training_data)
    return training_data

#prediction data processing
def process_test_data(prediction_dir,class_name):
    testing_data = []
    for img in tqdm(os.listdir(prediction_dir)):
        path = os.path.join(prediction_dir, img)
        if (img != 'Thumbs.db'):
            img_num = img.split('.')[0]
            image = Image.open(path)
            image = image.resize((IMG_SIZE, IMG_SIZE))
            testing_data.append([np.array(image), img_num])

    shuffle(testing_data)
    if (class_name == 'cucumber'):
        np.save('prediction_cucumber.npy', testing_data)
    elif (class_name == 'grape'):
        np.save('prediction_grape.npy', testing_data)
    elif (class_name == 'potato'):
        np.save('prediction_potato.npy', testing_data)
    return testing_data

#Training cucumber model
def train_cucumber_model():
    MODEL_NAME = 'plant_disease_identification_cucumber-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(convnet, tensorboard_dir='log',tensorboard_verbose=0)
    # Uncomment below block for Training cucumber model
    
    train_dir = '.\\train_data_cucumber'
    #if you need to create the data:
    train_data = create_train_data(train_dir, 'cucumber')
    #if you already have some saved:
    train_data = np.load('train_data_cucumber.npy')
    print('Training data loaded...')
    train = train_data[:-500]
    test = train_data[-500:]
    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    Y = [i[1] for i in train]
    test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    test_y = [i[1] for i in test]
    model.fit({'input': X}, {'targets': Y}, n_epoch=15, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=5000, show_metric=True, run_id=MODEL_NAME)
    model.save(MODEL_NAME)

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('Cucumber disease prediction model loaded!')

    import matplotlib.pyplot as plt
    # if you need to create the data:
    prediction_dir='.\\prediction_data_cucumber'
    test_data = process_test_data(prediction_dir,'cucumber')
    # if you already have some saved:
    test_data = np.load('prediction_cucumber.npy')
    print("Prediction data loaded...")
    with open('result_cucumber.csv', 'w') as f:
        f.write('Id,Probablity,Label\n')
    fig=plt.figure()
    with open('result_cucumber.csv', 'a') as f:
        for num,data in enumerate(test_data[:12]):
            img_num = data[1]
            img_data = data[0]
            y = fig.add_subplot(3,4,num+1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE,IMG_SIZE,3)
            model_out = model.predict([data])[0]
            if np.argmax(model_out) == 1: str_label='Healthy'
            else: str_label='Mildew Downy'
            f.write('{},{},{}\n'.format(img_num+'.jpeg', model_out[np.argmax(model_out)], str_label))
            y.imshow(np.squeeze(data))
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)
    plt.show()
