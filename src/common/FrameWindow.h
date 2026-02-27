#include <array>
#include <cassert>
#include <span>
#include <utility>

#include "frame.h"
#include "params.h"

class FrameWindow
{
public:
	static constexpr int W = mesh_vo::num_frames;

	FrameWindow(int width, int height)
		: latest_(width, height),
		  keyframe_(width, height), // requires Frame(width,height) ctor
		  window_(make_window_(width, height, std::make_index_sequence<W>{}))
	{
		// Start “empty”: you can also pre-fill by inserting first frames
		full_ = false;
		head_ = -1; // no frames yet
	}

	// Access the scratch frame where you write preprocessing output
	Frame &latest() { return latest_; }

	// After you compute pose/exposure/etc on latest, call this if you want to accept it.
	// Returns the index in the ring where it landed.
	int accept_latest()
	{
		const int insert_idx = next_insert_index_();
		std::swap(latest_, window_[insert_idx]);

		head_ = insert_idx;
		if (!full_ && count_ < W)
		{
			++count_;
			if (count_ == W)
				full_ = true;
		}
		return insert_idx;
	}

	// Number of valid frames currently stored (0..7)
	int size() const { return count_; }
	bool full() const { return full_; }

	// Oldest-to-newest view as pointers (so optimizer can modify frames)
	std::span<Frame *const> window_span_mut()
	{
		build_view_();
		return {view_.data(), static_cast<size_t>(count_)};
	}

	// Promote middle frame to keyframe (swap, no copies)
	void promote_middle_to_keyframe()
	{
		if (!full())
			return;
		int mid = (oldest_index_() + (W / 2)) % W; // for W=7 -> +3
		std::swap(keyframe_, window_[mid]);
	}

	Frame &keyframe_frame() { return keyframe_; }
	const Frame &keyframe_frame() const { return keyframe_; }

	// Middle frame (only meaningful once full)
	/*
	Frame &middle()
	{
		assert(full_ && "middle() requires full window");
		const int oldest = oldest_index_();
		const int mid = (oldest + 3) % W; // 0..6, middle of 7
		return window_[mid];
	}

	// Newest frame
	Frame &newest()
	{
		assert(count_ > 0);
		return window_[head_];
	}
	*/

private:
	template <std::size_t... Is>
	static std::array<Frame, W> make_window_(int w, int h, std::index_sequence<Is...>)
	{
		// (void)Is is just to expand the pack
		return {((void)Is, Frame(w, h))...};
	}

	int next_insert_index_() const
	{
		if (!full_)
		{
			// fill sequentially first
			return (head_ + 1) % W;
		}
		// when full, overwrite oldest
		return oldest_index_();
	}

	int oldest_index_() const
	{
		assert(count_ > 0);
		// oldest = next after head in a full ring
		if (full_)
			return (head_ + 1) % W;

		// not full: oldest is 0 (because we filled sequentially)
		return 0;
	}

	void build_view_()
	{
		// Build pointers oldest->newest into view_
		if (count_ == 0)
			return;

		int oldest = oldest_index_();
		for (int i = 0; i < count_; ++i)
		{
			int idx = (oldest + i) % W;
			view_[i] = &window_[idx];
		}
	}

private:
	Frame latest_;
	std::array<Frame, W> window_;
	Frame keyframe_;

	// Ring state
	int head_ = -1; // index of newest element in window_
	int count_ = 0; // number of valid frames (<=7)
	bool full_ = false;

	// View buffer (pointers oldest->newest)
	std::array<Frame *, W> view_{};
};