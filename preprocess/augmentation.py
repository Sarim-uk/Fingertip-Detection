import imgaug.augmenters as iaa
import random
import imgaug as ia
import numpy as np

def augment(train_x, train_y_prob, train_y_keys):
    aug = iaa.Sequential()
    x = random.uniform(-10, 10) * .01
    y = random.uniform(-10, 10) * .01
    aug.add(iaa.Affine(translate_percent={"x": x, "y": y},
                       scale=random.uniform(.7, 1.1),
                       rotate=random.uniform(-10, 10),
                       shear=random.uniform(-15, 15),
                       cval=(0, 255)))
    brightness = random.uniform(.5, 1.5)
    aug.add(iaa.Multiply(brightness))
    aug.add(iaa.CoarseDropout(p=.001, size_percent=0.005))
    aug.add(iaa.Dropout(p=(0, random.uniform(1, 5) * 0.005)))
    aug.add(iaa.Salt(.001))
    aug.add(iaa.AdditiveGaussianNoise(scale=random.uniform(.01, .1) * 255))
    pixel = 5
    aug.add(iaa.Crop(px=((0, random.randint(0, pixel)), (0, random.randint(0, pixel)),
                         (0, random.randint(0, pixel)), (0, random.randint(0, pixel)))))

    seq_det = aug.to_deterministic()
    image_aug = []
    keys_aug = []

    for i in range(0, train_x.shape[0]):
        image = train_x[i, :, :, :]
        prob = train_y_prob[i, :]
        keys = train_y_keys[i, :]

        image_aug.append(seq_det.augment_images([image])[0])
        koi = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]),
                                   ia.Keypoint(x=keys[2], y=keys[3]),
                                   ia.Keypoint(x=keys[4], y=keys[5]),
                                   ia.Keypoint(x=keys[6], y=keys[7]),
                                   ia.Keypoint(x=keys[8], y=keys[9])], shape=image.shape)

        k = seq_det.augment_keypoints([koi])[0]
        k = k.keypoints
        keys = [k[0].x, k[0].y, k[1].x, k[1].y, k[2].x, k[2].y, k[3].x, k[3].y, k[4].x, k[4].y]

        index = 0
        prob = prob.T[0][:-1]
        for j in range(0, len(prob)):
            keys[index] = keys[index] * prob[j]
            keys[index + 1] = keys[index + 1] * prob[j]
            index = index + 2
        keys_aug.append(keys)

    image_aug = np.asarray(image_aug)
    keys_aug = np.asarray(keys_aug)
    keys_aug = np.expand_dims(keys_aug, axis=-1)

    return image_aug, train_y_prob, keys_aug
