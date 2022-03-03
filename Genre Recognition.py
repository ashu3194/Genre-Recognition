import librosa.display
import matplotlib.pyplot as plt
import numpy
import sklearn
import pandas

FIG_SIZE = (15,10)

filename_brahms = "1.wav"
filename_busta = "2.wav"

# load audio file with Librosa
x_brahms, sr_brahms = librosa.load(filename_brahms, duration=30)
x_busta, sr_busta = librosa.load(filename_busta, duration=30)


# Plot the time-domain waveform of the audio signals:

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x_brahms, sr_brahms)

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x_busta, sr_busta)

# Compute the power melspectrogram:
S_brahms = librosa.feature.melspectrogram(x_brahms, sr=sr_brahms, power=2.0)

# Convert amplitude to decibels:
Sdb_brahms = librosa.power_to_db(S_brahms)

S_busta = librosa.feature.melspectrogram(x_busta, sr=sr_busta, power=2.0)
Sdb_busta = librosa.power_to_db(S_busta)

# plt.show()

# Extract Features

n_mfcc = 12
mfcc_brahms = librosa.feature.mfcc(x_brahms, sr=sr_brahms, n_mfcc=n_mfcc).T

print(mfcc_brahms.shape)
print(mfcc_brahms.mean(axis=0))

print(mfcc_brahms.std(axis=0))

# Scale Features
scaler = sklearn.preprocessing.StandardScaler()

mfcc_brahms_scaled = scaler.fit_transform(mfcc_brahms)
# is equivalent to:
#  scaler.fit(mfcc_brahms)
#  mfcc_brahms_scaled = scaler.transform(mfcc_brahms)

print(mfcc_brahms_scaled.mean(axis=0))
print(mfcc_brahms_scaled.std(axis=0))

# From the second audio file
mfcc_busta = librosa.feature.mfcc(x_busta, sr=sr_busta, n_mfcc=n_mfcc).T

print(mfcc_brahms.shape)
print(mfcc_busta.shape)

# Scaling
mfcc_busta_scaled = scaler.transform(mfcc_busta)

print(mfcc_busta_scaled.mean(axis=0))
print(mfcc_busta_scaled.std(axis=0))

# Train a Classifier
features = numpy.vstack((mfcc_brahms_scaled, mfcc_busta_scaled))
print(features.shape)

labels = numpy.concatenate((numpy.zeros(len(mfcc_brahms_scaled)), numpy.ones(len(mfcc_busta_scaled))))
print(labels.shape)

# Create a classifier model object:
model = sklearn.svm.SVC()

# Train the classifier
model.fit(features, labels)


# Run the classifier
x_brahms_test, sr_brahms = librosa.load(filename_brahms, duration=10, offset=120)
x_busta_test, sr_busta = librosa.load(filename_busta, duration=10, offset=120)


# Compute MFCCs from both of the test audio excerpts:
mfcc_brahms_test = librosa.feature.mfcc(x_brahms_test, sr=sr_brahms, n_mfcc=n_mfcc).T
mfcc_busta_test = librosa.feature.mfcc(x_busta_test, sr=sr_busta, n_mfcc=n_mfcc).T

print(mfcc_brahms_test.shape)
print(mfcc_busta_test.shape)

# Scale the MFCCs using the previous scaler:
mfcc_brahms_test_scaled = scaler.transform(mfcc_brahms_test)
mfcc_busta_test_scaled = scaler.transform(mfcc_busta_test)

# Concatenate all test features together:
features_test = numpy.vstack((mfcc_brahms_test_scaled, mfcc_busta_test_scaled))

# Concatenate all test labels together:
labels_test = numpy.concatenate((numpy.zeros(len(mfcc_brahms_test)), numpy.ones(len(mfcc_busta_test))))

# Compute the predicted labels:
predicted_labels = model.predict(features_test)

# Finally, compute the accuracy score of the classifier on the test data:
score = model.score(features_test, labels_test)
print(score)

predicted_labels = model.predict(mfcc_brahms_test_scaled)
print(predicted_labels.shape)

print(predicted_labels)

unique_labels, unique_counts = numpy.unique(predicted_labels, return_counts=True)
print(unique_labels, unique_counts)

# Analysis in Pandas
df_brahms = pandas.DataFrame(mfcc_brahms_test_scaled)

print(df_brahms.head())

df_busta = pandas.DataFrame(mfcc_busta_test_scaled)

print(df_brahms.corr())
print(df_busta.corr())

df_brahms.plot.scatter(1, 2, figsize=(7, 7))
df_busta.plot.scatter(1, 2, figsize=(7, 7))

df_brahms[0].plot.hist(bins=20, figsize=(14, 5))
df_busta[0].plot.hist(bins=20, figsize=(14, 5))

df_brahms[11].plot.hist(bins=20, figsize=(14, 5))
df_busta[11].plot.hist(bins=20, figsize=(14, 5))










