#pragma once
#include <zmq.hpp>

namespace super_ai {
    struct GameState_DAGGER;
    struct GameState_DQN;
}

namespace super_client
{
    extern zmq::context_t g_context;

    void close();

    class SuperClient {
    public:
        SuperClient(zmq::context_t& ctx);

        void init();
        void close() { socket.close(); }

        int request_action(const super_ai::GameState_DAGGER& state);
        int request_action(const super_ai::GameState_DAGGER& state, const int action);
        
        int request_action(const super_ai::GameState_DQN& state);

        int send_episode_score(int score);

        int set_learning_mode(bool mode);
    private:
        int request(void* msg, size_t len);

        zmq::context_t& ctx;
        zmq::socket_t socket;
    };
}
