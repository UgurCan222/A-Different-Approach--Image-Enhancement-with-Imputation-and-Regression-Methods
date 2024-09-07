import numpy as np
import cv2
import xgboost as xgb

input_path = r'C:\ImageAdjusterPixel\input.bmp'
output_path = r'C:\ImageAdjusterPixel\output.bmp'

image = cv2.imread(input_path, cv2.IMREAD_COLOR)
rows, cols, channels = image.shape

new_rows = 2 * rows - 1
new_cols = 2 * cols - 1
new_image = np.zeros((new_rows, new_cols, channels), dtype=np.uint8)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 100
}

def train_and_predict(X_train, y_train, X_test):
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model.predict(X_test)

for row in range(rows):
    for col in range(cols):
        new_image[2 * row, 2 * col, :] = image[row, col, :]

for row in range(rows):
    for col in range(cols):
        r_new, c_new = 2 * row, 2 * col
        
        # Estimate horizontal empty pixels (space to the right of it)
        if c_new + 2 < new_cols:
            for ch in range(channels):
                X_train = np.array([[image[row, col, ch], image[row, col + 1, ch]]])
                y_train = np.array([image[row, col + 1, ch]])
                predicted_value = train_and_predict(X_train, y_train, np.array([[image[row, col, ch], image[row, col + 1, ch]]]))
                new_image[r_new, c_new + 1, ch] = np.clip(predicted_value, 0, 255).astype(np.uint8)
        
        # Estimate vertical empty pixels (space below)
        if r_new + 2 < new_rows:
            for ch in range(channels):
                X_train = np.array([[image[row, col, ch], image[row + 1, col, ch]]])
                y_train = np.array([image[row + 1, col, ch]])
                predicted_value = train_and_predict(X_train, y_train, np.array([[image[row, col, ch], image[row + 1, col, ch]]]))
                new_image[r_new + 1, c_new, ch] = np.clip(predicted_value, 0, 255).astype(np.uint8)
        
        # Predict diagonal empty pixels (right-bottom space)
        if r_new + 2 < new_rows and c_new + 2 < new_cols:
            for ch in range(channels):
                X_train = np.array([[image[row, col, ch], image[row + 1, col, ch], image[row, col + 1, ch], image[row + 1, col + 1, ch]]])
                y_train = np.array([image[row + 1, col + 1, ch]])
                predicted_value = train_and_predict(X_train, y_train, X_train)
                new_image[r_new + 1, c_new + 1, ch] = np.clip(predicted_value, 0, 255).astype(np.uint8)

cv2.imwrite(output_path, new_image)
print(f"DONE")
