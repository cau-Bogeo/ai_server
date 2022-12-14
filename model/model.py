from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re

class AI:
    def combination(img_dir1, img_dir2):
        image1 = Image.open(img_dir1)
        image2 = Image.open(img_dir2)
        w1, h1 = image1.size
        w2, h2 = image2.size
        if w1 > h1:
            deg_image1 = image1.transpose(Image.ROTATE_90)
        else:
            deg_image1 = image1

        if w2 > h2:
            deg_image2 = image2.transpose(Image.ROTATE_90)
        else:
            deg_image2 = image2


        image1_size = deg_image1.size
        image2_size = deg_image2.size
        new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
        new_image.paste(deg_image1,(0,0))
        new_image.paste(deg_image2,(image1_size[0],0))
        new_image.save("merged_image.jpg","JPEG")
        #new_image.show()
    def test(img_dir):
        class_list= ['더블자임정','더블자임정2','더블자임정3','더블자임정4','러지피드정','러지피드정2','러지피드정3','러지피드정4']
        model = tf.keras.models.load_model('/home/ubuntu/ai_server/model/Pill_image_model_2_fix_5_2.h5')
        image = Image.open(img_dir)
        image = image.resize((224, 224))
        image = np.array(image)
        image = image/255.

        plt.imshow(image)
        plt.show()

        image = np.reshape(image, (1, 224, 224, 3))

        prediction = model.predict(image)
        #prediction.shape
        pred_class = np.argmax(prediction, axis=-1)
        #pred_class
        
        new_str = re.sub(r"[0-9]", "", class_list[int(pred_class)])
        return new_str