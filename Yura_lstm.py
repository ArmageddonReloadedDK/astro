#%%

# Как обычно импортируем все, что нужно.
# Из необычного тут только sklearn - полезная вещь,
# на нее тоже есть хорошая документация.
# Нам конкретно оттуда надо train_test_split.
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import numpy as np
import collections

# Очень полезная вещь для работы с текстовыми вещами.
# Детальное описание на ее сайте.
# При первом запуске надо раскоментить следующую строчку.
# Высплывет окошко и можно выбрать, что скачать.
# Можно не мучаться и скачать все.
# Для этого можно указать nltk.download("all").
import nltk
#nltk.download()

# Загрузка файла и исследоватеслький кусок кода.
# Хочется просто понять, сколько уникальных слов встречается
# в корпусе текстов и сколько в каждом предложениии.
# Что происходит: объявляем переменные, создают пустой счетчик,
# открываем файл и проходимся по каждой строчке файла.
# Для каждой строчки делаем две подстроки - метка и само предложение
# (они разделены в файле с помощью табуляции "\t").
# Далее делаем nltk.word_tokenize - классная вещь, игнорирует части речи,
# склонения и прочие параметры, которые для нашего демонстрационного
# примера не нужны.
# Ну а затем собствено обновляем наши переменные.
maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
ftrain = open("alice.txt", 'r')
for line in ftrain:
    label, sentence = line.strip().split("\t")
    words = nltk.word_tokenize(sentence.lower())
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    num_recs += 1
ftrain.close()

# Собственно вывыодим числа и смотрим на них.
print(maxlen)
print(len(word_freqs))

# Наши две ключевые константы.
# Мы задаем фиксированный размер словаря, а все остальные слова
# считаем несловарными и заменим впоследствии их фиктивноым словом UNK.
# Это позволит на этапе предсказания обрабатывать ранее не встречавшиеся
# слова как несловарные.
# Вторая константа задает фиксированную длину предложения и более
# короткие предложения будут дополнены нулями, а более длинные обрезаны.
# Формально можно это не делать, ибо РНС позволяет обрабатывать
# последовательности переменной длины, но мы для простоты сети
# будем обрабатывать предложения фиксированной длины.
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40

# Делаем таблицы соответствия.
# Входными данными для РНС является строка индексов слов, причем
# слова упорядочены по убыванию частоыт встречаемости в обучающем наборе
# (это важно, когда-нибудь покажу почему).
# Таблицы соответствия позволяют находить индекс по слову
# и слово по индексу (включая фиктивные слова PAD и UNK).
# 1 is UNK, 0 is PAD
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in
                enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}

# Здесь мы преобразуем входные предложения в последотельности
# индексов слов, дополняя (или обрезая) их до нужной длины.
# Выходные вещи (метки) специальным образом обрабатывать не надо.
X = np.empty((num_recs, ), dtype = list)
y = np.zeros((num_recs, ))
i = 0
ftrain = open("D:/Data/train.txt", 'r', encoding = 'utf-8')
for line in ftrain:
    label, sentence = line.strip().split("\t")
    words = nltk.word_tokenize(sentence.lower())
    seqs = []
    for word in words:
        if word in word2index:
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    X[i] = seqs
    y[i] = int(label)
    i += 1
ftrain.close()
# Эта функция собственно делает предложения нужной длины.
X = sequence.pad_sequences(X, maxlen = MAX_SENTENCE_LENGTH)

# Этой функцией делаем разбивку всего набора данных на обучающий и
# тестовый в пропорции 80:20.
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                                test_size = 0.2, random_state = 42)

# Тупо вывпедем, чтобы глянуть.
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

# Констатны для сети, их назначение реально должно быть понятно.
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Строим саму модель.
# Входными данными являеться последостность индексов слов.
# Длина последовательности равна MAX_SENTENCE_LENGTH.
# Первому измерению тензора присваивается значение None, показывающее,
# что размер пакета (чило записей, загружаемых в сеть за один раз)
# в момент определения сети неизвестен. он будет задан на этапе
# выполнения с помощью параметра batch_size.
# Таким образом в предположении, что размер пакет пока неизвестен,
# входной тензор имеет форму (None, MAX_SENTENCE_LENGTH, 1).
# Такие тензоры подаются на вход слоя погружения размера EMBEDDING_SIZE,
# веса которого инициализированы небольшими случайными значениями и
# подлежат обучению. Этот слой преобразует выходной тензор к форме
# (None, MAX_SENTENCE_LENGTH, EMBEDDING_SIZE). Выход слоя погружения
# загружается в LSTM с длиной последовательности MAX_SENTENCE_LENGTH
# и размером выходного слоя HIDDEN_LAYER_SIZE. На выходе LSTM получается
# тензор формы (None, HIDDEN_LAYER_SIZE, MAX_SENTENCE_LENGTH).
# По умолчанию LSTM выводи единственный тензор формы (None, HIDDEN_LAYER_SIZE)
# в качестве результирующей последовательности. Он подается на вход плотного
# слоя с размером выхода 1 и сигмоидной функцией активации.
# Ух... Тяжело, наверное.
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE,
                    input_length = MAX_SENTENCE_LENGTH))
model.add(Dropout(0.2))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(1))
model.add(Activation("softmax"))

# Компилируем модель, ничего интересного и сложного.
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

# Обучаем модель.
history = model.fit(Xtrain, ytrain, batch_size = BATCH_SIZE,
                    epochs = NUM_EPOCHS,
                    validation_data = (Xtest, ytest))

# Тестируем модель.
score, acc = model.evaluate(Xtest, ytest, batch_size = BATCH_SIZE)
print("Test score: %.3f, accuracy: %.3f" % (score, acc))

# Играемся с моделью.
# Выбираем случайные предложения из тестового набора и печатаем
# предсказание РНС, реальную метку и само предложение.
for i in range(50):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,40)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0].tolist() if x != 0])
    print("%.0f\t%d\t%s" % (ypred, ylabel, sent))


#%%


