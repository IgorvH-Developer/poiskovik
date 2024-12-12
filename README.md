# poiskovik
Search system

## Готовые базы данных можно скачать по ссылке: 

Единая большая БД:
* БД с файлами из папок с AA по AZ: https://drive.google.com/file/d/1v0C2u1l1vNKOjVtdGc5pG0msd9dTC0EQ/view?usp=sharing 742414 элемента

Несколько маленьких БД, в которых хранятся те же документы:
* БД с фалами из папок AA, AB, AD: https://drive.google.com/file/d/1-_NuMrGIoIIqs2XtUZ9qK_4LQL-m6GL0/view?usp=sharing 744954 элемента
* БД с фалами из папок AC, AE, AF: https://drive.google.com/file/d/1IzvYGFhlzuVd9bmEsrGRA2mIWIzHuvBV/view?usp=sharing 80782 элемента
* БД с фалами из папок AG, AH, AI: https://drive.google.com/file/d/14X2v96_u3ytjwGaje3RUhQX3aMT8LaX8/view?usp=sharing 101628 элемента
* БД с фалами из папок AJ, AK, AL: https://drive.google.com/file/d/1NUBi3YrZRzgdcOX4tBpX5z1tG-tLH7OL/view?usp=sharing 98488 элемента
* БД с фалами из папок AM, AN, AO: https://drive.google.com/file/d/1LwfmhxRBfqf-YDg6kilqHaJMkIYChavY/view?usp=sharing 98653 элемента
* БД с фалами из папок AP, AQ, AR: https://drive.google.com/file/d/1x3kMJbmmqHImcRwZJz-JvswhpTvKzyf8/view?usp=sharing 95519 элемента
* БД с фалами из папок AS, AT, AU: https://drive.google.com/file/d/10gStaAhQWz3h0HM_AFlYHXsEuIi3zWXu/view?usp=sharing 83965 элемента
* БД с фалами из папок AW, AV, AX: https://drive.google.com/file/d/1sHd6nMWp-919gNwuXTpwrN2SjdL1yaLl/view?usp=sharing 45838 элемента
* БД с фалами из папок AY, AZ: https://drive.google.com/file/d/188V9v1-DWiSH6fR01e5bRWkDkTFC27jP/view?usp=sharing 62592 элемента

Под базами данных имеются ввиду две БД: векторная БД, БД с метаданными.
Базы данных хранят документы из википедии, а конкретнее, документы разбиваются на блоки в соответствии с оглавлением документов.
Каждый блок хранится в виде вектрора в вектрной БД и в виде метаданных во второй БД.


### Векторная БД - vectorDB.index
- Вектор блока текста
- Индекс вектора

#### Принципы выбора индекса
https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

### БД с метаданными 

#### SQL БД - documentsMetadataDB.db (основной вариант)
- index
- url
- title
- proc_article

#### texts.csv (старый вариант)
- id
- url
- текст блока
- обработанный текст блока

*Порядок записей в вектроной БД и БД с метаданными синхронизирован*

## Тестовый скрипт (test_models.py)

В папке data/queries_split лежат статьи из википедии, разбитые на абзацы, к каждому из которых придуман вопрос. Тестовая система берёт один такой файл, запускает поисковик на каждом из вопросов, а затем измеряет, как быстро были получены результаты и на каком месте был "правильный" абзац.

Запуск: python3 test_models.py test_configs.txt
Скрипт перебирает все возможные конфигурации из файла test_configs.txt и выводит оценку, среднее время и стандартное отклонение времени.

Параметры конфигурации:

Имя датасета --- скрипт добавляет к нему путь и '.txt' и берёт данные из соответствующего файла
ranker --- ранжирующая модель (None, MB25 или CrossEncoder). Если указано None, ранжирование происходит с помощью векторной базы.
doc_num --- количество документов, которые выдаёт векторная база. Произвольное натуральное число
metric --- метрика для оценки качества (пока только inv_5). Оценивает качество списка абзацев по тому, на какой позиции встретился размеченный. inv_5 считается по формуле $\frac{5}{n + 5}$

Результаты тестирования:

| Параметры поисковика | Литва, score | Литва, t | Лесков, score | Лесков, t | Метро, score | Метро, t| Перестройка, score | Перестройка, t |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
|  None, doc_nun = 3   |    0.537     |  0.175   |     0.217     |   0.188   |    0.533     |  0.174  |                    |                |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
|  None, doc_nun = 5   |    0.498     |  0.180   |     0.233     |   0.183   |    0.521     |  0.170  |                    |                |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
|  None, doc_nun = 10  |    0.423     |  0.178   |     0.183     |   0.181   |    0.573     |  0.179  |                    |                |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
|  None, doc_nun = 20  |    0.418     |  0.179   |     0.207     |   0.173   |    0.383     |  0.177  |                    |                |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
|  None, doc_nun = 50  |    0.267     |  0.176   |     0.083     |   0.178   |    0.237     |  0.178  |                    |                |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
|  BM25, doc_nun = 3   |    0.514     |  0.171   |     0.199     |   0.177   |    0.567     |  0.178  |                    |                |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
|  BM25, doc_nun = 5   |    0.485     |  0.189   |     0.222     |   0.173   |    0.593     |  0.179  |                    |                |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
|  BM25, doc_nun = 10  |    0.496     |  0.202   |     0.181     |   0.172   |    0.726     |  0.177  |                    |                |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
|  BM25, doc_nun = 20  |    0.470     |  0.175   |     0.237     |   0.171   |    0.626     |  0.179  |                    |                |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
|  BM25, doc_nun = 50  |    0.448     |  0.181   |     0.181     |   0.184   |    0.537     |  0.173  |                    |                |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
|  CEnc, doc_nun = 3   |    0.578     |  0.840   |     0.233     |   0.821   |    0.602     |  0.573  |                    |                |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
|  CEnc, doc_nun = 5   |    0.599     |  1.025   |     0.277     |   1.251   |    0.651     |  0.900  |                    |                |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
|  CEnc, doc_nun = 10  |    0.695     |  2.120   |     0.261     |   2.376   |    0.838     |  1.736  |                    |                |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
|  CEnc, doc_nun = 20  |    0.739     |  4.135   |     0.339     |   4.574   |    0.827     |  3.683  |                    |                |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
|  CEnc, doc_nun = 50  |    0.757     |  10.110  |     0.324     |   11.093  |    0.799     |  7.662  |                    |                |
|----------------------|--------------|----------|---------------|-----------|--------------|---------|--------------------|----------------|
>>>>>>> Stashed changes


Нужно добавить:
1) Кэширование страницы. То есть, чтобы при повторном одинаковом запросе (как GET, так и POST) не проводился поиск, а выдавалась сохранённая страница
2) Оценка релевантности запроса (кворум), чтоб не выдавать мусор.
3) Суммаризацию доделать
4) И структурировать и кластеризовать выдачу (то есть если несколько блоков из одной страниы, то сначал показать страницу, а потом пункты)
5) Исследование по векторному индексу. Надо чтобы проверили разные конфигурации ANN и остановились на лучший (а не сразу на Flatt). Добавить latency, пропускная способность, throuput. И понять что можно улучшить.
6) По ранжированию подумать о би энкодере, а на BM25 можно подумать что потюнить (коэффы, proximity, нормализация, стэмминг).
