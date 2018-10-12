import dataset
import network

book_data, book_char, K = dataset.read_data()

net = network.RNN(K, K, book_char)

net.fit(book_data)
# net.test_gradient(book_data[:net.seq_length], book_data[1:net.seq_length+1])
