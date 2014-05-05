#!/usr/bin/env python3

import pandas as pd

def wiki_tables(dictionary, header, title='Attribute name'):
    output = '{{|border="1"\n!{0}\n!'.format(title)
    output += '\n!'.join(header)
    for attr in sorted(dictionary):
        output += '\n|-\n|{0}'.format(attr)
        for field in header:
            output += '\n|'
            if field in dictionary[attr]:
                if field.startswith('count'): #HACK
                    output += '{0}'.format(int(dictionary[attr][field]))
                else:
                    output += '{:.2f}'.format(dictionary[attr][field])
            else:
                output += "''n/a''"
    output += '\n|}'

    return output

def main():
    emotions = pd.read_json('data/emotions.json')
    landmarks = pd.read_json('data/landmarks.json')
    facs = pd.read_json('data/facs.json')

    header = ['min', 'max', 'mean', 'std', 'count']

    print('===Emotions===')
    emotions_dict = emotions[['emotion']].describe().to_dict()
    emotions_dict['emotion']['count (incl. null values)'] = len(emotions)
    print(wiki_tables(emotions_dict, header+['count (incl. null values)']))

    print('===Landmarks===')
    print(wiki_tables(landmarks[['x', 'y']].describe().to_dict(), header))

    print('===FACS===')
    print("====Summary====")
    print(wiki_tables(facs[['au', 'intensity']].describe().to_dict(), header))

    print("====Details for each AU====")
    grouped_facs = facs.groupby(['au'])['intensity'].describe().to_dict()
    facs_dict = {}
    for au, field in grouped_facs:
        if not au in facs_dict:
            facs_dict[au] = {}

        facs_dict[au][field] = grouped_facs[(au, field)]

    print(wiki_tables(facs_dict, header, title='AU'))

if __name__ == '__main__':
    main()
