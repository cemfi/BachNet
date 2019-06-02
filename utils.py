
def generate_txt_output(data, path):
    with open(path, 'w') as fp:
        for pitches in data.t().flip(dims=[0]):
            line = ''
            for step in pitches:
                char = '*' if step == 1 else ' '
                line += char
            fp.write(line)
            fp.write('\n')