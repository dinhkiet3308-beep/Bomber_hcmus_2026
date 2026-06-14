import random
from collections import deque
import heapq


class TacticalRuleAgent:
    """
    Agent Quan – Improved tactical agent.

    Key improvements over baseline tactical_rule_agent:
    1. Chain-reaction-aware danger model with per-tile detonation timers
    2. Time-aware BFS pathfinding (can walk through tiles that explode later)
    3. Weighted bomb-placement scoring (enemy hits, boxes, escape quality)
    4. Dead-end / corridor awareness to avoid traps
    5. Game-phase strategy: aggressive early, survival-first late
    6. Center-gravity fallback + active enemy hunting
    """

    MOVES = {
        0: (0, 0),
        1: (-1, 0),   # UP
        2: (1, 0),     # DOWN
        3: (0, -1),    # LEFT
        4: (0, 1),     # RIGHT
    }
    team_id = "Agent Quan"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step_count = 0

    # ==================================================================
    # MAIN DECISION LOOP
    # ==================================================================

    def act(self, obs):
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]
        self.step_count += 1

        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            return 0

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        bomb_radius = max(1, int(bomb_bonus) + 1)
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}

        enemies = [
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and p[2] == 1
        ]
        enemy_set = set(enemies)
        blocked = set(bomb_positions)
        blocked.discard(my_pos)

        danger_times = self._build_danger_map(grid, bombs, players)
        phase = "early" if self.step_count <= 120 else ("mid" if self.step_count <= 350 else "late")

        # ============================================================
        # PRIORITY 0: ESCAPE FROM DANGER
        # ============================================================
        my_dangers = danger_times.get(my_pos, set())
        need_escape = bool(my_dangers)

        if need_escape:
            escape = self._escape(grid, my_pos, blocked, danger_times, enemy_set)
            if escape is not None:
                return escape
            # Ult desperation fallback: any valid move not exploding at step 1
            valid_moves = self._valid_actions(grid, my_pos, blocked, enemy_set)
            safe_moves = [a for a in valid_moves if not self._is_tile_dangerous_at(danger_times, self._next_pos(my_pos, a), 1)]
            return random.choice(safe_moves) if safe_moves else 0

        # ============================================================
        # PRIORITY 1: COLLECT NEARBY ITEMS (Global Vision)
        # ============================================================
        item_tiles = self._item_tiles(
            grid, prefer_capacity=int(bombs_left) <= 1, prefer_radius=int(bomb_bonus) <= 1
        )
        if item_tiles:
            safe_item_tiles = set()
            for item in item_tiles:
                if self._manhattan(my_pos, item) <= 1:
                    safe_item_tiles.add(item)
                else:
                    if self._open_neighbors(grid, item, blocked) >= 2:
                        safe_item_tiles.add(item)

            if safe_item_tiles:
                # Removed the short-sighted max_dist cap. Search the whole map.
                move = self._move_to_targets(grid, my_pos, safe_item_tiles, blocked, danger_times, enemy_set, max_dist=20)
                if move is not None:
                    return move

        # ============================================================
        # PRIORITY 2: PLACE BOMB (if valuable + safe)
        # ============================================================
        if bombs_left > 0 and my_pos not in bomb_positions:
            bomb_action = self._consider_bombing(
                grid, my_pos, enemies, bomb_radius, blocked, danger_times, phase, enemy_set, obs
            )
            if bomb_action is not None:
                return bomb_action

        # ============================================================
        # PRIORITY 3: MOVE TO BOMB SPOT (farm boxes - NEVER STOP)
        # ============================================================
        box_spots = self._ranked_box_spots(grid, my_pos, blocked, bomb_radius)
        if box_spots:
            # Removed the phase=="late" lock. Always farm boxes to win tiebreakers!
            move = self._move_to_targets(grid, my_pos, box_spots, blocked, danger_times, enemy_set, max_dist=20)
            if move is not None:
                return move

        # ============================================================
        # PRIORITY 4: ATTACK ENEMIES (Relentless Pressure)
        # ============================================================
        attack_spots = self._attack_positions(grid, enemies, blocked, bomb_radius)
        if attack_spots:
            # Removed the phase=="late" lock. Always hunt enemies!
            move = self._move_to_targets(grid, my_pos, attack_spots, blocked, danger_times, enemy_set, max_dist=20)
            if move is not None:
                return move

        # ============================================================
        # PRIORITY 5: STAT BOMB (tiebreaker padding)
        # ============================================================
        is_board_empty = not box_spots  # True if all boxes are destroyed
        
        if (self.step_count >= 250 or is_board_empty): 
            if bombs_left > 0 and my_pos not in bomb_positions:
                if self._can_escape_after_placing(grid, my_pos, blocked, danger_times, bomb_radius, enemy_set):
                    my_exits = self._open_neighbors(grid, my_pos, blocked)
                    # We require >= 3 exits here so we only stat-bomb in wide open areas,
                    # ensuring we never accidentally trap ourselves in a corridor just for a point.
                    if my_exits >= 3:  
                        return 5

        # ============================================================
        # PRIORITY 6: STRATEGIC FALLBACK
        # ============================================================
        return self._strategic_fallback(grid, my_pos, enemies, blocked, danger_times, phase, enemy_set)

    # ==================================================================
    # GEOMETRY HELPERS
    # ==================================================================

    def _next_pos(self, pos, action):
        dx, dy = self.MOVES[action]
        return pos[0] + dx, pos[1] + dy

    def _in_bounds(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and grid[x, y] in (0, 3, 4)

    def _valid_actions(self, grid, my_pos, blocked, enemy_set=None):
        actions = [0]
        for a in (1, 2, 3, 4):
            nx, ny = self._next_pos(my_pos, a)
            if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                if enemy_set is None or (nx, ny) not in enemy_set:
                    actions.append(a)
        return actions

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_adjacent_to_enemy(self, pos, enemy_set):
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (pos[0] + dx, pos[1] + dy) in enemy_set:
                return True
        return False

    def _open_neighbors(self, grid, pos, blocked):
        cnt = 0
        for a in (1, 2, 3, 4):
            nx, ny = self._next_pos(pos, a)
            if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                cnt += 1
        return cnt

    # ==================================================================
    # BLAST / DANGER MODEL
    # ==================================================================

    def _blast_tiles(self, grid, bx, by, radius):
        """Return set of tiles hit by a bomb at (bx,by) with given radius."""
        tiles = {(bx, by)}
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for r in range(1, radius + 1):
                x, y = bx + dx * r, by + dy * r
                if not self._in_bounds(grid, x, y):
                    break
                cell = grid[x, y]
                if cell == 1:  # wall
                    break
                tiles.add((x, y))
                if cell == 2:  # box — blast stops after hitting it
                    break
        return tiles

    def _build_danger_map(self, grid, bombs, players):
        """
        Build a per-tile danger map: danger_times[tile] = set of steps at which
        the tile will be hit by an explosion. Tracks ALL explosion times, not
        just the earliest, so the agent won't walk into a later bomb after
        dodging an earlier one.

        Returns dict: (x, y) -> set of int (each int = ticks until explosion)
        """
        if len(bombs) == 0:
            return {}

        # Parse bomb info
        bomb_info = []
        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner_id = int(b[3]) if len(b) > 3 else -1
            # Always use actual owner's radius when available
            radius = 2  # fallback
            if 0 <= owner_id < len(players):
                radius = max(1, int(players[owner_id][4]) + 1)
            bomb_info.append({
                "pos": (bx, by),
                "timer": timer,
                "radius": radius,
            })

        # Chain reaction propagation: if bomb A's blast hits bomb B,
        # bomb B detonates at the same time as A (min timer)
        changed = True
        while changed:
            changed = False
            for i, ba in enumerate(bomb_info):
                blast_a = self._blast_tiles(grid, ba["pos"][0], ba["pos"][1], ba["radius"])
                for j, bb in enumerate(bomb_info):
                    if i == j:
                        continue
                    if bb["pos"] in blast_a and bb["timer"] > ba["timer"]:
                        bb["timer"] = ba["timer"]
                        changed = True

        # Build the danger map — store ALL explosion times per tile
        danger_times = {}
        for bomb in bomb_info:
            blast = self._blast_tiles(grid, bomb["pos"][0], bomb["pos"][1], bomb["radius"])
            for tile in blast:
                if tile not in danger_times:
                    danger_times[tile] = set()
                danger_times[tile].add(bomb["timer"])

        return danger_times

    def _is_tile_dangerous_at(self, danger_times, pos, step):
        """Check if a tile has an explosion at exactly this step."""
        times = danger_times.get(pos, set())
        return step in times

    def _earliest_danger(self, danger_times, pos):
        """Get the earliest explosion time for a tile, or 999 if safe."""
        times = danger_times.get(pos, set())
        return min(times) if times else 999

    def _any_danger_in_range(self, danger_times, pos, from_step, to_step):
        """Check if any explosion hits the tile between from_step and to_step (inclusive)."""
        times = danger_times.get(pos, set())
        return any(from_step <= t <= to_step for t in times)

    # ==================================================================
    # ESCAPE SYSTEM
    # ==================================================================

    def _escape(self, grid, my_pos, blocked, danger_times, enemy_set):
        """
        Dijkstra-based escape. Finds the safest escape route, penalizing
        enemy proximity to avoid body-blocking. If no perfectly safe tile is
        reachable, it gracefully degrades to find the tile that explodes latest.
        """
        # pq elements: (cost, dist, pos, first_action)
        # We want to minimize cost.
        pq = []
        seen = {my_pos: 0}

        for a in (1, 2, 3, 4):
            nx, ny = self._next_pos(my_pos, a)
            npos = (nx, ny)
            if not self._passable(grid, nx, ny):
                continue
            if npos in blocked:  # bomb positions
                continue
            if npos in enemy_set:  # hard-blocked at step 1
                continue
            if self._is_tile_dangerous_at(danger_times, npos, 1):
                continue

            # Penalty for step 1
            step_penalty = 6 if self._is_adjacent_to_enemy(npos, enemy_set) else 0
            seen[npos] = 1 + step_penalty
            heapq.heappush(pq, (1 + step_penalty, 1, npos, a))

        best_action = None
        best_score = -10**9

        while pq:
            cost, dist, pos, first_action = heapq.heappop(pq)

            # Check survival time at this tile
            tile_times = danger_times.get(pos, set())
            future_dangers = [t for t in tile_times if t >= dist]
            earliest_future = min(future_dangers) if future_dangers else 999

            if earliest_future > dist:
                # We can survive here for at least one step!
                # Score it:
                score = earliest_future * 1000 + self._open_neighbors(grid, pos, blocked) * 10 - cost
                if score > best_score:
                    best_score = score
                    best_action = first_action

            if dist >= 8:
                continue

            for a in (1, 2, 3, 4):
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                next_dist = dist + 1
                if not self._passable(grid, nx, ny):
                    continue
                if npos in blocked:
                    continue
                if self._is_tile_dangerous_at(danger_times, npos, next_dist):
                    continue

                step_penalty = 0
                if npos in enemy_set:
                    step_penalty = 15
                elif self._is_adjacent_to_enemy(npos, enemy_set):
                    step_penalty = 6

                next_cost = cost + 1 + step_penalty

                if npos in seen and seen[npos] <= next_cost:
                    continue
                seen[npos] = next_cost
                heapq.heappush(pq, (next_cost, next_dist, npos, first_action))

        return best_action

    # ==================================================================
    # PATHFINDING (Time-aware BFS)
    # ==================================================================

    def _move_to_targets(self, grid, start, targets, blocked, danger_times, enemy_set, max_dist=15):
        """
        Dijkstra-based pathfinding to find the safest and shortest path to targets.
        Avoids stepping next to enemies (soft danger) and prevents stepping on enemies at step 1.
        """
        if not targets:
            return None

        # pq elements: (cost, dist, pos, first_action)
        pq = [(0, 0, start, None)]
        seen = {start: 0}  # pos -> cost

        while pq:
            cost, dist, pos, first_action = heapq.heappop(pq)

            if pos in targets and first_action is not None:
                return first_action

            if dist >= max_dist:
                continue

            for a in (1, 2, 3, 4):
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                next_dist = dist + 1

                if not self._passable(grid, nx, ny):
                    continue
                # Bombs are always blocked
                if npos in blocked and npos not in targets:
                    continue
                # Enemies are hard-blocked at step 1 (to avoid immediate collision)
                if next_dist == 1 and npos in enemy_set and npos not in targets:
                    continue

                # Danger check: completely avoid active blast zones
                if npos in danger_times:
                    continue

                # Calculate cost/penalty for this step
                step_penalty = 0
                if npos in enemy_set:
                    step_penalty = 15  # high penalty for stepping on enemy's current tile
                elif self._is_adjacent_to_enemy(npos, enemy_set):
                    step_penalty = 6   # soft danger zone penalty

                next_cost = cost + 1 + step_penalty

                if npos in seen and seen[npos] <= next_cost:
                    continue

                seen[npos] = next_cost
                heapq.heappush(pq, (next_cost, next_dist, npos, a if first_action is None else first_action))

        return None

    # ==================================================================
    # BOMBING INTELLIGENCE
    # ==================================================================

    def _consider_bombing(self, grid, my_pos, enemies, bomb_radius, blocked, danger_times, phase, enemy_set, obs):
        """
        Decide whether to place a bomb. Uses weighted scoring:
        - 5 points per enemy in blast
        - 1 point per box in blast
        - Trap bonus for cornered enemies
        Late game: still bombs (threshold=1) to accumulate tiebreaker stats.
        """
        # Check what we'd hit
        my_blast = self._blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)

        enemy_hits = 0
        for ex, ey in enemies:
            if (ex, ey) in my_blast:
                enemy_hits += 1

        boxes_hit = sum(1 for x, y in my_blast if grid[x, y] == 2)

        # Can enemy be trapped? (enemy in blast zone with limited escape)
        trap_bonus = 0
        for ex, ey in enemies:
            if (ex, ey) in my_blast:
                enemy_exits = self._open_neighbors(grid, (ex, ey), blocked | {my_pos})
                if enemy_exits <= 1:
                    trap_bonus += 3  # high value: enemy likely trapped

        # Score the bomb
        score = enemy_hits * 5 + boxes_hit * 1 + trap_bonus

        # Threshold is always 1 — we want to bomb whenever there's value,
        # including late game (tiebreaker: boxes destroyed, bombs placed)
        if score < 1:
            return None

        # Can we escape after placing?
        if not self._can_escape_after_placing(grid, my_pos, blocked, danger_times, bomb_radius, enemy_set, obs):
            return None

        # Dead-end check: don't bomb if we're in a corridor with <=1 exit
        my_exits = self._open_neighbors(grid, my_pos, blocked)
        if my_exits <= 1:
            return None

        return 5  # place bomb

    def _can_bomb_hit_enemy(self, grid, my_pos, enemies, radius):
        """Check if a bomb at my_pos would hit any enemy."""
        blast = self._blast_tiles(grid, my_pos[0], my_pos[1], radius)
        for ex, ey in enemies:
            if (ex, ey) in blast:
                return True
        return False

    def _can_escape_after_placing(self, grid, my_pos, blocked, danger_times, bomb_radius, enemy_set, obs):
        """
        Simulate placing a bomb and check if we can escape the blast.
        Uses _build_danger_map to simulate the exact future danger map.
        """
        # Create simulated bombs list by adding our prospective bomb
        sim_bombs = list(obs["bombs"]) + [[my_pos[0], my_pos[1], 7, self.agent_id]]
        sim_danger = self._build_danger_map(grid, sim_bombs, obs["players"])

        # Detonation time of our bomb
        my_bomb_timer = min(sim_danger.get(my_pos, {7}))

        # Hard-stop if dropping the bomb triggers an instant suicide
        if my_bomb_timer <= 1:
            return False

        my_blast = self._blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
        max_escape_steps = max(1, my_bomb_timer - 1)

        # BFS escape simulation
        q = deque([(my_pos, 0)])
        seen = {my_pos: 0}

        while q:
            pos, dist = q.popleft()

            if pos not in my_blast and dist > 0:
                # Verify destination is safe from future explosions
                pos_dangers = sim_danger.get(pos, set())
                future = {t for t in pos_dangers if t >= dist}
                if not future:
                    return True

            if dist >= max_escape_steps:
                continue

            for a in (1, 2, 3, 4):
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                next_dist = dist + 1

                if not self._passable(grid, nx, ny):
                    continue
                if npos in blocked:
                    continue
                if next_dist == 1 and npos in enemy_set:
                    continue
                if next_dist in sim_danger.get(npos, set()):
                    continue
                if npos in seen and seen[npos] <= next_dist:
                    continue

                seen[npos] = next_dist
                q.append((npos, next_dist))

        return False

    # ==================================================================
    # ATTACK POSITIONING
    # ==================================================================

    def _attack_positions(self, grid, enemies, blocked, radius):
        """
        Find tiles from which we could bomb an enemy.
        Prioritizes spots where the enemy is cornered, but will fall back 
        to open-field pressure if no traps are available.
        """
        best_spots = set()
        normal_spots = set()
        
        for ex, ey in enemies:
            enemy_exits = self._open_neighbors(grid, (ex, ey), blocked)

            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                for r in range(1, radius + 1):
                    x, y = ex + dx * r, ey + dy * r
                    
                    if not self._in_bounds(grid, x, y):
                        break
                    if not self._passable(grid, x, y):
                        break
                    if (x, y) in blocked:
                        continue
                        
                    # Verify line of sight
                    if self._line_clear(grid, (x, y), (ex, ey)):
                        # Prefer positions with >= 2 exits (so we can escape our own bomb)
                        if self._open_neighbors(grid, (x, y), blocked) >= 2:
                            if enemy_exits <= 2:
                                best_spots.add((x, y)) # Trap spot
                            else:
                                normal_spots.add((x, y)) # Pressure spot
                                
        # Always return the highest priority targets available
        return best_spots if best_spots else normal_spots

    def _line_clear(self, grid, a, b):
        """Check if there's a clear line of sight between two tiles (same row/col)."""
        ax, ay = a
        bx, by = b
        if ax == bx:
            step = 1 if by > ay else -1
            for y in range(ay + step, by, step):
                if grid[ax, y] in (1, 2):
                    return False
            return True
        if ay == by:
            step = 1 if bx > ax else -1
            for x in range(ax + step, bx, step):
                if grid[x, ay] in (1, 2):
                    return False
            return True
        return False

    # ==================================================================
    # BOX FARMING
    # ==================================================================

    def _ranked_box_spots(self, grid, my_pos, blocked, bomb_radius):
        """
        Find spots next to boxes where bombing is valuable.
        Rank by number of boxes that would be hit.
        """
        spots = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y] != 2:
                    continue
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = x + dx, y + dy
                    if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                        spots.add((nx, ny))

        if not spots:
            return set()

        # Rank by boxes we'd hit from each spot — keep only the best ones
        scored = []
        for sx, sy in spots:
            blast = self._blast_tiles(grid, sx, sy, bomb_radius)
            boxes = sum(1 for bx, by in blast if grid[bx, by] == 2)
            exits = self._open_neighbors(grid, (sx, sy), blocked)
            if exits >= 2:  # don't suggest spots we can't escape from
                scored.append(((sx, sy), boxes))

        if not scored:
            return spots  # fallback to unranked

        # Keep spots that hit >= max(1, best-1) boxes
        scored.sort(key=lambda t: -t[1])
        best_boxes = scored[0][1]
        threshold = max(1, best_boxes - 1)
        return {pos for pos, b in scored if b >= threshold}

    # ==================================================================
    # ITEM COLLECTION
    # ==================================================================

    def _item_tiles(self, grid, prefer_capacity=False, prefer_radius=False):
        preferred_values = set()
        if prefer_radius:
            preferred_values.add(3)
        if prefer_capacity:
            preferred_values.add(4)

        preferred_tiles = {
            (x, y)
            for x in range(grid.shape[0])
            for y in range(grid.shape[1])
            if grid[x, y] in preferred_values
        }
        if preferred_tiles:
            return preferred_tiles

        return {
            (x, y)
            for x in range(grid.shape[0])
            for y in range(grid.shape[1])
            if grid[x, y] in (3, 4)
        }

    # ==================================================================
    # STRATEGIC FALLBACK
    # ==================================================================

    def _find_center_target(self, grid):
        """Find the nearest passable tile to the map center.
        The exact center (6,6) on a 13x13 grid is always a wall (checkerboard)."""
        cx, cy = grid.shape[0] // 2, grid.shape[1] // 2
        if self._passable(grid, cx, cy):
            return (cx, cy)
        # Spiral outward to find nearest passable tile
        for radius in range(1, 5):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # only border of square
                        nx, ny = cx + dx, cy + dy
                        if self._passable(grid, nx, ny):
                            return (nx, ny)
        return (cx, cy)  # ultimate fallback

    def _strategic_fallback(self, grid, my_pos, enemies, blocked, danger_times, phase, enemy_set):
        """
        When no specific objective, move strategically:
        - Late game: stay near center, avoid danger
        - Mid game: drift toward enemies or center
        - Early game: explore toward center
        """
        valid = self._valid_actions(grid, my_pos, blocked, enemy_set)

        # Filter to safe actions: no explosion at step 1 or 2
        safe_actions = []
        for a in valid:
            npos = self._next_pos(my_pos, a)
            if not self._is_tile_dangerous_at(danger_times, npos, 1) and \
               not self._is_tile_dangerous_at(danger_times, npos, 2):
                safe_actions.append(a)

        if not safe_actions:
            return 0

        # Remove "stay" if we have movement options (avoid camping)
        move_actions = [a for a in safe_actions if a != 0]

        if not move_actions:
            return 0  # stay is the only option

        # Find a passable center tile (not a wall)
        center = self._find_center_target(grid)

        # In mid/late game, if enemies are nearby, drift toward them for kills
        if phase != "late" and enemies:
            closest_enemy = min(enemies, key=lambda e: self._manhattan(my_pos, e))
            if self._manhattan(my_pos, closest_enemy) <= 6:
                target = closest_enemy
            else:
                target = center
        else:
            target = center

        # Pick action that moves us closest to target
        best_action = None
        best_dist = 999
        best_open = -1

        for a in move_actions:
            npos = self._next_pos(my_pos, a)
            dist = self._manhattan(npos, target)
            open_n = self._open_neighbors(grid, npos, blocked)
            # Prefer: closer to target, then more open neighbors
            if dist < best_dist or (dist == best_dist and open_n > best_open):
                best_dist = dist
                best_open = open_n
                best_action = a

        return best_action if best_action is not None else random.choice(safe_actions)
