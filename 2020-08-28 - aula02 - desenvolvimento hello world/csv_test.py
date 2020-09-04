import csv

def raw_data_gen(n):
    '''
    generator for mock data


    yields str generators
    '''
    for i in range(n):
        yield (f'{i}_{j}' for j in range(4))


#create/overwirte a file with rawdata
with open('data_file.csv', 'w', newline='') as data_buffer:
    file_writer = csv.writer(data_buffer)
    file_writer.writerows(raw_data_gen(5))


#reads a file with rawdata and prints it
with open('data_file.csv', 'r', newline='') as data_buffer:
    file_reader = csv.reader(data_buffer)
    for row in file_reader:
        print(row)

