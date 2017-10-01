import csv

with open('resources/timings.csv', 'rb') as timings:
    timing_csv = csv.reader(timings)
    labels = []
    start_times = []
    end_times = []
    for row in timing_csv:
        sl = [float(v) for v in row[0].split(':')]
        el = [float(v) for v in row[2].split(':')]
        label = '_'.join(row[1].split(' '))
        start_time = (sl[0] - 1) * 3600 + sl[1] * 60 + sl[2] + sl[3] * 1 / 25.0
        dur = el[0] * 3600 + el[1] * 60 + el[2] + el[3] * 1 / 25.0
        end_time = start_time+dur
        labels.append(label)
        start_times.append(start_time)
        end_times.append(end_time)

    with open('resources/cutting_times.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i, j, k in zip(start_times, end_times, labels):
            writer.writerow([i, j, k])

