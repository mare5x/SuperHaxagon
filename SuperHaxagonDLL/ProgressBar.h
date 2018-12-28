#pragma once
#include <cstdio>
#include <cmath>

/* A utility class for displaying a simple progress bar in the command line. */
class ProgressBar {
public:
	ProgressBar(int total, int width = 32, char bar_char = '#') : bar_char(bar_char), width(width), current(0), total(total) 
	{ 
		int c = total;
		total_len = 0;
		while (c) {
			c /= 10;
			++total_len;
		}
	}

	void update()
	{
		++current;

		printf("%*d/%d -- [", total_len, current, total);
		int filled = round((double)current / total * width);
		for (int i = 0; i < filled; ++i)
			putchar(bar_char);
		for (int i = 0; i < (width - filled); ++i)
			putchar(' ');
		printf("]\r");
	}

	void reset()
	{
		current = 0;
	}

	void clear()
	{
		int len = 0;
		len += total_len * 2 + 1 + 4 + 1;
		len += width;
		len += 1;
		for (int i = 0; i < len; ++i)
			printf(" ");
		printf("\r");
	}

private:
	char bar_char;
	int width;
	int current;
	int total;
	int total_len;
};