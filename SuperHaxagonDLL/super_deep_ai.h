#pragma once

struct SuperStruct;

namespace dqn_ai {
	struct GameState {
		float walls[6][2];  // distance and width
		float player_pos;
		float player_slot;
		float wall_speed;
	};

	void init(bool load = true);

	void exit(bool save = true);

	// Query the server to find out in which direction the AI should move.
	int get_move_dir(SuperStruct* super, bool learning);

	// Call this when training the agent and it dies.
	void report_death(SuperStruct* super);

    // Respond to requests made from the client process.
    void respond_server();
}