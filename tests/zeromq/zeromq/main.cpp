#include <string>
#include <iostream>

#include <zmq.hpp>


enum MSG_TYPE {
    ACTION, DEATH
};

struct action_t {
    int32_t type{ ACTION };
    int32_t action = -1;
    double dummy_state[9] = { .0, .1, .2, .3, .4, 1.0, 2.0, 3.0, 4.0 };
};

struct death_t {
    int32_t type{ DEATH };
    int32_t data = 42;
};

int main()
{
    zmq::context_t context{ 1 };

    zmq::socket_t socket{ context, zmq::socket_type::req };
    socket.connect("tcp://localhost:5555");

    action_t dummy_action;
    death_t dummy_death;
    zmq::const_buffer action_buf = zmq::buffer(&dummy_action, sizeof(action_t));
    zmq::const_buffer death_buf = zmq::buffer(&dummy_death, sizeof(dummy_death));

    for (int i = 0; ; ++i) {
        // std::cout << "Sending: " << i << std::endl;
        // socket.send(zmq::buffer(&i, sizeof(int)), zmq::send_flags::none);
        if (i % 2 == 0) {
            std::cout << "Sending: action" << std::endl;
            socket.send(action_buf);
        }
        else {
            std::cout << "Sending: death\n";
            socket.send(death_buf);
        }

        zmq::message_t reply{};
        socket.recv(reply, zmq::recv_flags::none);
        // std::cout << reply.to_string() << std::endl;
        std::cout << *static_cast<int*>(reply.data()) << std::endl;
    }

    return 0;
}