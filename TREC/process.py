import os

def process(source, save_dir):
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
        open(os.path.join(save_dir, intent), 'a').write('\n'.join(sentencelist))
process('1000', 'dev')
process('2000', 'train')
process('3000', 'train')
process('5500', 'train')
process('t', 'test')