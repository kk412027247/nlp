import csv
import json

valcaptions = json.load(open('../data/annotations/captions_val2014.json', 'r'))
trcaptions = json.load(open('../data/annotations/captions_train2014.json', 'r'))

print(trcaptions.keys())

prefix = '../data/'
val_prefix = prefix + 'val2014/'
train_prefix = prefix + 'train2014/'

trimages = {x['id']: x['file_name'] for x in trcaptions['images']}
valset = len(valcaptions['images']) - 5000

valimages = {x['id']: x['file_name'] for x in valcaptions['images'][:valset]}
truevalimg = {x['id']: x['file_name'] for x in valcaptions['images'][valset:]}

data = list()
errors = list()
validation = list()

for item in trcaptions['annotations']:
    if int(item['image_id']) in trimages:
        fpath = train_prefix + trimages[int(item['image_id'])]
        caption = item['caption']
        data.append((caption, fpath))
    else:
        errors.append(item)

for item in valcaptions['annotations']:
    caption = item['caption']
    if int(item['image_id']) in valimages:
        fpath = val_prefix + valimages[int(item['image_id'])]
        data.append((caption, fpath))
    elif int(item['image_id']) in truevalimg:
        fpath = val_prefix + truevalimg[int(item['image_id'])]
        validation.append((caption, fpath))
    else:
        errors.append(item)

print(errors)

with open(prefix + 'data.csv', 'w') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    writer.writerow(data)

with open(prefix + 'validation.csv', 'w') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)

print('TRAINING: Total Number of Captions: {}, Total Number of Images: {}'.format(
    len(data), len(trimages) + len(valimages)))

print('VALIDATING/TESTING Total Number of Captions:{}, Total Number of Images: {}'.format(
    len(validation), len(truevalimg)))

print('Errors: ', errors)
