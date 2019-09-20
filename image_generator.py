from keras.preprocessing import image
import os

src = 'backgrounds/'

datagen = image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=180,
    width_shift_range=[-30, 30],
    height_shift_range=[-30, 30],
    shear_range=0.,
    # zoom_range=[-3, 3],
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=None,)

gen_data = datagen.flow_from_directory(src,
                                   batch_size=1,
                                   shuffle=False,
                                   save_to_dir=os.path.join('gen', src),
                                   save_prefix='gen',
                                   target_size=(100, 100))
for i in range(1000):
    gen_data.next()
