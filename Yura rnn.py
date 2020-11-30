# %%

# классика жанра - импортируем минимально необходимые модули
import numpy as np
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.preprocessing.sequence import sequence


# открываем файл для чтения побайтово и создаем пустой массив
# в этом массив попадут строки из файла
fin = open("D:/Data/алиса2.txt", 'rb')
lines = []

# проходимся по каждой строчке внутри файла
# каждый символ переводим в нижний регистра
# все "лишние" символы и пустые строки игнорируем
for line in fin:
    line = line.strip().lower()
    line = line.decode("utf-8", "ignore")
    if len(line) == 0:
        continue
    lines.append(line)
fin.close()

# объединили все строки в один большой текст
text = " ".join(lines)

# ну и напечатали его
print(text)

# %%

# поскольку рассматриваемая здесь РНС будет предсказывать символы,
# то и словарь будет состояить из множества символов, встречающихся в тексте
# собственно эти 4 строчки создают два словаря:
# 1. как по символу получить его индекс
# 2. как по индексу получить соответствующий индекс
chars = set([c for c in text])
nb_chars = len(chars)
char2index = dict((c, i) for i, c in enumerate(chars))
index2char = dict((i, c) for i, c in enumerate(chars))

# просто выведем число уникальных символов в тексте
print(nb_chars)

# %%

# указываем, что проходим по тексту с шагом в STEP символов
# и выделеям отрезки длиной SEQLEN
STEP = 1
SEQLEN = 10

# собственно делаем проход по тексту
# делаем для каждого отрезка длиной SEQLEN метку в один символ
# Например для текста "The sky was falling" будет
# "The sky wa" -> s
# "he sky was" -> ' '
# "e sky was " -> f (в этой строке в конце пробел!)
# " sky was f" -> a (а в этой строке в начале пробел)
input_chars = []
label_chars = []
for i in range(0, len(text) - SEQLEN, STEP):
    input_chars.append(text[i:i + SEQLEN])
    label_chars.append(text[i + SEQLEN])

# здесь происходит векторизация входных строк и меток
# по сути подготавливаем матрицы для обучения РНС
# на вход РНС подаются построенные выше входные строки
# в каждой из них SEQLEN символов, а поскольку размер нашего словаря
# составляет nb_chars символов, то каждый входной символ представляеся
# унитарный вектором длины nb_chars
# следовательно каждый входной элемент представляет собой тензор формы
# SEQLEN * nb_chars
# выходныа метка - это единственный символ, поэтому по аналогии
# с представлением входных символов она представляется унитарным
# вектором длины nb_chars
X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)
Y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)
for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        X[i, j, char2index[ch]] = 1
    Y[i, char2index[label_chars[i]]] = 1

# базовые константы для нашей РНС
HIDDEN_SIZE = 128
BATCH_SIZE = 128
NUM_ITERATIONS = 25
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 100

# собственно создаем обычную простую РНС
# как всегда указываем последовательную модель
# далее указываем простой слой РНС с рядом параметров
# (все параметры хорошо описаны в документации к керасу)
# например нам важен параметр return_sequences - в нашем случае
# нужно получить на выходе всего один символ, а не последовательность
# после РНС указывается обычный денсовый слой
# ну и функция активации softmax, поскольку хотим получать на выходе вероятность
model = Sequential()
model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=False,
                    input_shape=(SEQLEN, nb_chars),
                    unroll=True))
model.add(Dense(nb_chars))
model.add(Activation("softmax"))

# компилируем модель, указываем функцию потерь и оптимизатор
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

# здесь собственно демонстрация как модель работает
# самая тяжелая часть :)
# раньше мы всегда обучали сеть в течение фиксированного числа периодов,
# а затем оценивали ее на зарезервированных тестовых данных
# поскольку в данном случае у нас нет помеченных данных, то мы
# выполняем один период обучения (NUM_EPOCHS_PER_ITERATION = 1), а затем
# тестируем модель
# так происходит на протяжении NUM_ITERATIONS раз
# следовательно, по существу мы выполняем NUM_ITERATIONS периодов обучения
# и тестируем модель после каждого периода
# вот :)
# тестирование производится так: модель порождает символ по заднным
# входным данным, затем первый символ входной строки отбрасывается,
# в конец дописывается предсказанный на предыдущем прогоне символ и
# у модели запрашивается следующее предсказание
# так повторяется NUM_PREDS_PER_EPOCH раз, после чего
# полученная строка печается
# собственно эта строка и является индикатором качества модели
# надеюсь, что все понятно :)
for iteration in range(NUM_ITERATIONS):
    print("=" * 50)  # просто красиво разделяю строки
    print("Iteration #: %d" % (iteration))

    model.fit(X, Y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)

    # выбираем произвольный индекс в тексте и оттуда берем входные символы
    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]
    print("Generating from seed: %s" % (test_chars))
    print(test_chars, end="")

    # собственно само предсказание как описано выше
    for i in range(NUM_PREDS_PER_EPOCH):
        xtest = np.zeros((1, SEQLEN, nb_chars))
        for i, ch in enumerate(test_chars):
            xtest[0, i, char2index[ch]] = 1

        pred = model.predict(xtest, verbose=0)[0]
        ypred = index2char[np.argmax(pred)]

        print(ypred, end="")
        test_chars = test_chars[1:] + ypred
    print()

# %%


