import numpy
start_range = 1
end_range = 7-2
bin_range = numpy.linspace(start_range,
        end_range, end_range-start_range+1)\
                .astype('int').tolist()

print(numpy.linspace(0, 1, 100))