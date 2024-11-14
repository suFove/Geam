import csv
import json

from utils.common import split_dataset


def mergeFile(f1, f2, f3):
    texts = []
    labels = []
    with open(f1, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in data:
            label, text = line.split('\t')
            texts.append(text.replace('\r', '').strip())
            labels.append(int(label))
        f.close()

    with open(f2, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in data:
            label, text = line.split('\t')
            texts.append(text.replace('\r', '').strip())
            labels.append(int(label))
        f.close()

    with open(f3, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in data:
            label, text = line.split('\t')
            texts.append(text.replace('\r', '').strip())
            labels.append(int(label))
        f.close()

    return texts, labels


def list2json(texts, labels, fname):
    json_list = []
    for text, label in zip(texts, labels):
        json_list.append({"label": label, "text": text})
    with open(fname, 'w', encoding='utf') as f:
        json.dump(json_list, f, ensure_ascii=False, indent=1)

def list2csv(texts, labels, fname):
    with open(fname, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['text', 'label'])
        for text, label in zip(texts, labels):
            # 每一行写入文本和标签
            writer.writerow([text, label])


def json2csv(srcfname, outfname):
    json_data = []
    with open(srcfname, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    print(json_data[0]['text'])
    print(json_data[0]['label'])
    with open(outfname, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['label','text'])
        # for text in json_data:
        for j_data in json_data:
            text = j_data['text']
            label = j_data['label']
            # 每一行写入文本和标签
            writer.writerow([label, text])


if __name__ == '__main__':
    train_path = '../mydatasets/TCM_SD/train.txt'
    dev_path = '../mydatasets/TCM_SDdev.txt'
    test_path = '../mydatasets/TCM_SD/test.txt'
    json_path = '../mydatasets/TCM_SD/data.json'
    csv_path = '../mydatasets/TCM_SD/data.csv'

    # texts, labels = mergeFile(train_path, dev_path, test_path)

    # print(len(texts))
    # print(len(labels))
    # print(texts[0])
    # print(labels[0])
    # list2json(texts, labels, json_path)
    # json2csv(json_path, csv_path)

    split_dataset(csv_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)