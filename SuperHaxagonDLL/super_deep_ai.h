#pragma once

struct SuperStruct;

namespace dqn_ai {
	void init(bool load = true);

	void exit(bool save = true);

	// Query the ANN to find out in which direction the ai should move.
	int get_move_dir(SuperStruct* super, bool learning);

	// Call this when training the agent and it dies.
	void report_death(SuperStruct* super);
}