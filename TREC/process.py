import os
def process(source, train_dir, dev_dir):
    result = {}
    for line in open(source,'r').readlines():
        line = line.strip().split(' ')
        intent = line[0].split(':')
        intent = intent[0]#+'-'+intent[1]
        line.pop(0)
        line.pop(len(line)-1)
        result.setdefault(intent, [])
        result[intent].append(' '.join(line))
    for intent, sentencelist in result.items():
        intent_len = len(sentencelist)
        train_len = int(intent_len*0.9)
        train_list = sentencelist[0:train_len]
        dev_list = sentencelist[train_len:intent_len]
        open(os.path.join(train_dir, intent), 'w').write('\n'.join(train_list))
        open(os.path.join(dev_dir, intent), 'w').write('\n'.join(dev_list))
process('5000', 'train', 'dev')
