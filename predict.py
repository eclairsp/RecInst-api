from keras.models import Sequential
from keras.models import load_model
import json

model = load_model("cnn-med-model-backtothefututre.h5")

def predict():
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img("spectrograms/193480f61cdb40078ea7e3e6fecdb97f-gaccla0518__1.png", target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict_proba(test_image)
    op = {
        'gac' : result[0][0].item(),
        'gel' : result[0][1].item(),
        'org' : result[0][2].item(),
        'pia' : result[0][3].item(),
        'voi' : result[0][4].item()
    } # '.item() helps with result number being float32 and cant be used in json'
    return op

res = predict()
print(res)