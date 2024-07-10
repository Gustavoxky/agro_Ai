# -*- coding: utf-8 -*-
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window
import pandas as pd
import fiona
from fiona.crs import from_epsg
from shapely.geometry import shape, mapping, box, Point, Polygon, LineString, MultiPolygon
from sklearn.model_selection import train_test_split
from rasterio.plot import show
from pylab import rcParams
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam  # Importar Adam corretamente
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to the image and shapefiles
path_img = 'AOI_img_rep.tif'
path_classe1 = 'Classe_1.shp'
path_classe2 = 'Classe_2.shp'
path_classe3 = 'Classe_3.shp'

# Function to read shapefiles using fiona
def read_shapefile(path):
    logging.info(f'Reading shapefile: {path}')
    with fiona.open(path, 'r') as src:
        return [feature for feature in src]

# Read shapefiles
logging.info('Reading shapefiles...')
gdf1 = read_shapefile(path_classe1)
gdf2 = read_shapefile(path_classe2)
gdf3 = read_shapefile(path_classe3)

# Plot image and shapefiles
fig, ax = plt.subplots(figsize=(20, 20))
with rasterio.open(path_img) as src:
    crs = src.crs
    show(src, ax=ax)
    
    # Convert shapefile geometries to the same CRS as the image
    def plot_shapefile(shapes, color, ax):
        for feature in shapes:
            geom = shape(feature['geometry'])
            if isinstance(geom, Point):
                # Handle Point geometries differently
                logging.warning('Skipping Point geometry in shapefile plot.')
                continue
            elif isinstance(geom, (Polygon, LineString, MultiPolygon)):
                geom = geom.to_crs(crs.to_dict())
                x, y = geom.xy
                ax.plot(x, y, color=color)
            else:
                logging.warning(f'Skipping unsupported geometry type: {type(geom)}')
                continue

    logging.info('Plotting shapefiles on image...')
    plot_shapefile(gdf1, 'red', ax)
    plot_shapefile(gdf2, 'yellow', ax)
    plot_shapefile(gdf3, 'blue', ax)

# Open image with rasterio
src = rasterio.open(path_img)
im = src.read().transpose([1, 2, 0]).astype('uint8')

# Assign IDs to shapefile features
for feature in gdf1:
    feature['properties']['id'] = 0
for feature in gdf2:
    feature['properties']['id'] = 1
for feature in gdf3:
    feature['properties']['id'] = 2

# Combine all shapefiles into one
gdf = gdf1 + gdf2 + gdf3

# Create image and label lists
img_list = []
label_list = []
for feature in gdf:
    geom = shape(feature['geometry'])
    x, y = geom.xy[0][0], geom.xy[1][0]
    label = feature['properties']['id']
    row, col = src.index(x, y)
    img_patch = im[row-64:row+64, col-64:col+64, 0:3]
    
    img_list.append(img_patch)
    label_list.append(label)

X = np.array(img_list)
Y = np.array(label_list)

# Plot sample image
dict_name = {0: 'Daninha_tipo_1', 1: 'Daninha_tipo_2', 2: 'Normal'}
i = 20
plt.figure(figsize=[6, 6])
plt.title(dict_name[Y[i]])
plt.imshow(X[i])
plt.axis('off')

# Normalize images
X = X / 255

# One-hot encode labels
Y = Y[:, np.newaxis]
enc = OneHotEncoder()
enc.fit(Y)
Y = enc.transform(Y).toarray()

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), kernel_initializer="he_normal", padding='same', input_shape=(x_train.shape[1:])),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(64, (3, 3), kernel_initializer="he_normal", padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(128, (3, 3), kernel_initializer="he_normal", padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(256, (3, 3), kernel_initializer="he_normal", padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dropout(0.5),
    Dense(512),
    Activation('relu'),
    Dense(3),
    Activation('softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.00001), metrics=['accuracy'])

# Print model summary
logging.info('Model summary:')
model.summary()

# Train the model
logging.info('Training the model...')
history = model.fit(x_train, y_train, batch_size=64, epochs=200, verbose=1, shuffle=True, validation_split=0.2)

# Plot training history
plt.style.use('fivethirtyeight')
rcParams['figure.figsize'] = 18, 6

# Accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()

# Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

# Evaluate model
logging.info('Evaluating the model...')
predict = model.predict(x_test)
pred = np.argmax(predict, axis=1)
true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(true, pred)
logging.info(f'Accuracy: {accuracy}')
print(classification_report(true, pred))

# Confusion matrix
cm = confusion_matrix(true, pred)
logging.info(f'Confusion matrix:\n{cm}')

# Confusion matrix heatmap
class_list = ['Daninha_tipo_1', 'Daninha_tipo_2', 'Normal']
columns = class_list
r1 = pd.DataFrame(data=cm, columns=columns, index=columns)
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(r1, annot=True, annot_kws={"size": 18}, fmt='d', cmap="inferno_r", cbar=False)
ax.tick_params(labelsize=16)
ax.set_ylabel('Verdadeiro')
ax.set_xlabel('Predito')

# Split image into patches and classify
path_img_to_pred = 'AOI_img_rep.tif'
path_split = "split_img"
if not os.path.isdir(path_split):
    os.mkdir(path_split)

src = rasterio.open(path_img_to_pred)
out_meta = src.meta.copy()
qtd = 0

for n in range((src.meta['width'] // 128)):
    for m in range((src.meta['height'] // 128)):
        x = (n * 128)
        y = (m * 128)
        window = Window(x, y, 128, 128)
        win_transform = src.window_transform(window)
        arr_win = src.read(window=window)
        
        if arr_win.max() != 0:
            qtd += 1
            path_exp_img = os.path.join(path_split, f'img_{qtd}.tif')
            out_meta.update({"driver": "GTiff", "height": arr_win.shape[1], "width": arr_win.shape[2], "compress": 'lzw', "transform": win_transform})
            with rasterio.open(path_exp_img, 'w', **out_meta) as dst:
                for i, layer in enumerate(arr_win, start=1):
                    dst.write_band(i, layer.reshape(-1, layer.shape[-1]))
            logging.info(f'Created image: {qtd}')
        del arr_win

# Predict on patches
n = [f for f in os.listdir(path_split)]
df_full = []

for path_img in n:
    path_full = os.path.join(path_split, path_img)
    ds = rasterio.open(path_full, 'r')
    im = ds.read().transpose([1, 2, 0])[:, :, :3] / 255
    im = im[np.newaxis, :, :, :]
    predict = model.predict(im)
    predict = np.argmax(predict, axis=1)[0]
    geom = box(*ds.bounds)
    df_full.append({"id": 1, "classe": predict, "geometry": mapping(geom)})

# Write predictions to shapefile
schema = {'geometry': 'Polygon', 'properties': {'id': 'int', 'classe': 'int'}}
crs = from_epsg(4326)

with fiona.open('veg_classification.shp', 'w', driver='ESRI Shapefile', schema=schema, crs=crs) as dst:
    for record in df_full:
        dst.write(record)

# Plot classified patches
ax = plt.gca()
df_full_gpd = gpd.GeoDataFrame(df_full).set_crs(crs)
df_full_gpd.plot(column='classe', legend=True, cmap='RdYlGn', categorical=True, legend_kwds={"loc": "center left", "bbox_to_anchor": (1, 0.5)}, ax=ax)

# Function to replace legend items
def replace_legend_items(legend, mapping):
    for txt in legend.texts:
        for k, v in mapping.items():
            if txt.get_text() == str(k):
                txt.set_text(v)

replace_legend_items(ax.get_legend(), dict_name)

plt.show()
