#include "stdafx.h"
#include "super_client.h"
#include "super_ai.h"


namespace {
    enum MSG_TYPE {
        DAGGER_STATE_ACTION = 3,
        DAGGER_STATE_EXPERT_ACTION = 4,
        EPISODE_SCORE = 5,
        DQN_STATE_ACTION = 6,
        DQN_LEARNING_MODE = 7
    };
}


namespace super_client {
    zmq::context_t g_context{ 1 };

    SuperClient::SuperClient(zmq::context_t& _ctx) 
        : ctx(_ctx), socket{ _ctx, zmq::socket_type::req } { }

    void SuperClient::init()
    {
        socket.connect("tcp://localhost:5555");
    }

    int SuperClient::request_action(const super_ai::GameState_DAGGER & state)
    {
        struct StateAction_t {
            int32_t type;
            super_ai::GameState_DAGGER state;
        } msg{ DAGGER_STATE_ACTION, state };

        //printf("REQ: request action\n");
        int reply = request(&msg, sizeof(StateAction_t));
        //printf("REP: %d\n", reply);
        return reply;
    }

    int SuperClient::request_action(const super_ai::GameState_DAGGER & state, const int action)
    {
        struct StateActionPair_t {
            int32_t type;
            int32_t action;
            super_ai::GameState_DAGGER state;
        } msg{ DAGGER_STATE_EXPERT_ACTION, action, state };

        //printf("REQ: request action w/ demonstrator\n");
        int reply = request(&msg, sizeof(StateActionPair_t));
        //printf("REP: %d\n", reply);
        return reply;
    }

    int SuperClient::request_action(const super_ai::GameState_DQN & state)
    {
        struct StateAction_t {
            int32_t type;
            super_ai::GameState_DQN state;
        } msg{ DQN_STATE_ACTION, state };

        //printf("REQ: request action\n");
        int reply = request(&msg, sizeof(StateAction_t));
        //printf("REP: %d\n", reply);
        return reply;
    }

    int SuperClient::send_episode_score(int score)
    {
        struct EpisodeScore_t {
            int32_t type;
            int32_t score;
        } msg{ EPISODE_SCORE, score };

        printf("REQ: episode score (%d)\n", score);
        int reply = request(&msg, sizeof(EpisodeScore_t));
        printf("REP: %d\n", reply);
        return reply;
    }

    int SuperClient::set_learning_mode(bool mode)
    {
        struct Msg_t {
            int32_t type;
            int32_t mode;
        } msg{ DQN_LEARNING_MODE, mode };

        printf("REQ: learning mode (%d)\n", mode);
        int reply = request(&msg, sizeof(Msg_t));
        printf("REP: %d\n", reply);
        return reply;
    }

    int SuperClient::request(void * msg, size_t len)
    {
        zmq::const_buffer buf = zmq::buffer(msg, len);
        socket.send(buf);

        zmq::message_t reply{};
        socket.recv(reply, zmq::recv_flags::none);

        return *reply.data<int>();
    }

    void close()
    {
        g_context.close();
    }
}
