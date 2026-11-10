# C simulator core (no CUDA). Build: make test_load && ./test_load baked/<x>.sbb
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L
INC = -I.
LDLIBS = -lm

CORE = sb_baked.c sb_chol.c sb_lu.c sb_fdm.c sb_rng.c sb_convect.c sb_occup.c sb_sim.c sb_reward.c sb_obs.c

.PHONY: all clean
all: test_load test_chol test_lu_big test_fdm test_step test_multi test_batch test_step_batch test_episode libsb_bridge.so

test_load: test/test_load.c $(CORE)
	$(CC) $(CFLAGS) $(INC) test/test_load.c $(CORE) -o test_load $(LDLIBS)

test_chol: test/test_chol.c $(CORE)
	$(CC) $(CFLAGS) $(INC) test/test_chol.c $(CORE) -o test_chol $(LDLIBS)

test_lu_big: test/test_lu_big.c $(CORE)
	$(CC) $(CFLAGS) $(INC) test/test_lu_big.c $(CORE) -o test_lu_big $(LDLIBS)

test_fdm: test/test_fdm.c $(CORE)
	$(CC) $(CFLAGS) $(INC) test/test_fdm.c $(CORE) -o test_fdm $(LDLIBS)

clean:
	rm -f test_load test_chol test_lu_big test_fdm test_chol_big \
	      test_step test_multi test_batch test_step_batch test_episode \
	      libsb_bridge.so

test_step: test/test_step.c $(CORE)
	$(CC) $(CFLAGS) $(INC) test/test_step.c $(CORE) -o test_step $(LDLIBS)

test_multi: test/test_multi.c $(CORE)
	$(CC) $(CFLAGS) $(INC) test/test_multi.c $(CORE) -o test_multi $(LDLIBS)

test_batch: test/test_batch.c $(CORE)
	$(CC) $(CFLAGS) $(INC) test/test_batch.c $(CORE) -o test_batch $(LDLIBS)

test_step_batch: test/test_step_batch.c $(CORE)
	$(CC) $(CFLAGS) $(INC) test/test_step_batch.c $(CORE) -o test_step_batch $(LDLIBS)

test_episode: test/test_episode.c $(CORE)
	$(CC) $(CFLAGS) $(INC) test/test_episode.c $(CORE) -o test_episode $(LDLIBS)

libsb_bridge.so: sb_bridge.c $(CORE)
	$(CC) $(CFLAGS) -fPIC -shared $(INC) sb_bridge.c $(CORE) -o libsb_bridge.so $(LDLIBS)

test_episodes: test/test_episodes.c $(CORE)
	$(CC) $(CFLAGS) $(INC) test/test_episodes.c $(CORE) -o test_episodes $(LDLIBS)
