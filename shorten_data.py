import random

new_data = []
counter = 0
prev = 0
new_val_data = []
other_counter = 0
with open('train.tsv') as f:
    for line in f:
        counter += 1
        if random.random() < 0.06250:
            other_counter += 1
            if other_counter % 2 == 0:
                new_data.append(line)
            else:
                new_val_data.append(line)
            if counter - prev > 1000000:
                print('Read:', counter)
                prev = counter

print(f'Saving {len(new_data)} lines out of {counter}')
open('val_short.tsv', 'w').write(''.join(new_val_data))
open('train_short.tsv', 'w').write('\n'.join(new_data))
