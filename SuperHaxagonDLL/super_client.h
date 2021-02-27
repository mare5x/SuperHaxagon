#pragma once
#include <zmq.hpp>
#include "super_deep_ai.h"

struct dqn_ai::GameState;

namespace super_client
{
    extern zmq::context_t g_context;

    class SuperClient {
    public:
        SuperClient(zmq::context_t& ctx);

        void init();

        int request_action(const dqn_ai::GameState& state);
        int request_action(const dqn_ai::GameState& state, const int action);
        
        int send_episode_score(int score);
    private:
        int request(void* msg, size_t len);

        zmq::context_t& ctx;
        zmq::socket_t socket;
    };
}
