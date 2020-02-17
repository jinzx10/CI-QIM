#ifndef __WIDGETS_H__
#define __WIDGETS_H__

#include <iostream>
#include <chrono>
#include <string>
#include <cstdlib>

struct Stopwatch
{
	using iclock	= std::chrono::high_resolution_clock;
	using dur_t		= std::chrono::duration<double>;

	Stopwatch(): t_start(), dur_store(dur_t::zero()), is_running(false) {}

	void run() {
		if (is_running) {
			std::cout << "The stopwatch is already running. Nothing to do." 
				<< std::endl;
		} else {
			t_start = iclock::now();
			is_running = true;
		}
	}

	void pause() { 
		if (is_running) {
			dur_store += iclock::now() - t_start;
			is_running = false;
		} else {
			std::cout << "The stopwatch is not running. Nothing to do." 
				<< std::endl;
		}
	}

	void report() { 
		dur_t dur = is_running ? 
			dur_store + static_cast<dur_t>(iclock::now() - t_start) : dur_store;
		std::cout << "time elapsed = " << dur.count() << " seconds" 
			<< std::endl; 
	}

	void reset() { 
		dur_store = dur_store.zero();
		is_running = false;
	}

	iclock::time_point t_start;	
	dur_t dur_store;
	bool is_running;
};


inline int mkdir(std::string const& dir) {
	std::string command = "mkdir -p " + dir;
	return std::system(command.c_str());
}



#endif
