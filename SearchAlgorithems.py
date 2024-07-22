
from typing import TypeVar, Iterable, Sequence, Generic, List, Callable, Set, Deque, Dict, Any, Optional
from typing_extensions import Protocol
from heapq import heappush, heappop
from collections import deque



T = TypeVar('T')

def linear_contains(iterable: Iterable[T], key: T) -> bool:
    for item in iterable:
        if item == key:
            return True
    return False
    
class Node(Generic[T]):
    def __init__(self, state: T, parent: Optional['Node'], cost: float = 0.0, heuristic: float = 0.0, reverse_node: Optional['Node'] = None) -> None:
        self.state: T = state
        self.parent: Optional[Node] = parent
        self.cost: float = cost
        self.heuristic: float = heuristic
        self.reverse_node: Optional[Node] = reverse_node

    def __lt__(self, other: 'Node') -> bool:
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    def __eq__(self, other: 'Node') -> bool:
        return (self.cost + self.heuristic) == (other.cost + other.heuristic)

    def __repr__(self) -> str:
        return f"Node({self.state}, {self.parent}, {self.cost}, {self.heuristic}, {self.reverse_node})"
    
class Stack(Generic[T]):
    def __init__(self) -> None:
        self._container: List[T] = []
    @property
    def empty(self) -> bool:
        return not self._container

    def push(self, item: T) -> None:
        self._container.insert(0, item)

    def pop(self) -> T:
        return self._container.pop(0)

    def __repr__(self) -> str:
        return repr(self._container)
    

def node_to_path(Node: Node[T]) -> List[T]:
    path: List[T] = [Node.state]

    while Node.parent is not None:
        path.append(Node.parent.state)
        Node = Node.parent
    path.reverse()
    return path

class Queue(Generic[T]):
    def __init__(self) -> None:
        self._container: Deque[T] = []
    
    @property
    def empty(self) -> bool:
        return not self._container
    
    def push(self, item: T) -> None:
        self._container.append(item)
    
    def pop(self) -> T:
        return self._container.pop(0)
    
    def __repr__(self) -> str:
        return repr(self._container)
    


class PriorityQueue(Generic[T]):
    def __init__(self) -> None:
        self._container: List[T] = []
    
    @property
    def empty(self) -> bool:
        return not self._container
    
    def push(self, item: T) -> None:
        heappush(self._container, item)
    
    def pop(self) -> T:
        return heappop(self._container)
    
    def __repr__(self) -> str:
        return repr(self._container)
    


class PriorityQueueWithFunctions(PriorityQueue):

    def push(self, item, *args, **kwargs):
        super().put(item, *args, **kwargs)



def depthFirstSearch(initial: T, goal_test: Callable[[T], bool], successors: Callable[[T], List[T]]) -> Optional[Node[T]]:
    frontier: Stack[Node[T]] = Stack()
    frontier.push(Node(initial, None))
    explored: Set[T] = {initial}

    while not frontier.empty:
        current_node: Node[T] = frontier.pop()
        current_state: T = current_node.state

        if goal_test(current_state):
            print("Nodes visited:", len(explored))
            return current_node
        for child in successors(current_state):
            if child in explored:
                continue
            explored.add(child)
            frontier.push(Node(child, current_node, cost=1.0))
    print("Nodes visited:", len(explored))
    return None


 
def breadthFirstSearch(initial: T, goal_test: Callable[[T], bool], successors: Callable[[T], List[T]]) -> Optional[Node[T]]:
    frontier: Queue[Node[T]] = Queue()
    frontier.push(Node(initial, None))
    explored: Set[T] = {initial}

    while not frontier.empty:
        current_node: Node[T] = frontier.pop()
        current_state: T = current_node.state
        if goal_test(current_state):
            print("Nodes visited:", len(explored))
            return current_node
        sorted_successors = sorted(successors(current_state))
        for child in sorted_successors:
            if child in explored:
                continue
            explored.add(child)
            frontier.push(Node(child, current_node))
    print("Nodes visited:", len(explored))
    return None


    
def greedyBestFirst(initial: T, goal_test: Callable[[T], bool], successors: Callable[[T], List[T]], heuristic: Callable[[T], float]) -> Optional[Node[T]]:
    frontier: PriorityQueue[Node[T]] = PriorityQueue()
    frontier.push(Node(initial, None, 0.0, heuristic(initial)))
    explored: Set[T] = {initial}

    while not frontier.empty:
        current_node: Node[T] = frontier.pop()
        current_state: T = current_node.state
        if goal_test(current_state):
            print("Nodes visited:", len(explored))
            return current_node
        for child in successors(current_state):
            if child in explored:
                continue
            explored.add(child)
            frontier.push(Node(child, current_node, 0.0, heuristic(child)))
    print("Nodes visited:", len(explored))
    return None


def astar(initial: T, goal_test: Callable[[T], bool], successors: Callable[[T], List[T]], heuristic: Callable[[T], float]) -> Optional[Node[T]]:
    frontier: PriorityQueue[Node[T]] = PriorityQueue()
    frontier.push(Node(initial, None, 0.0, heuristic(initial)))
    explored: Dict[T, float] = {initial: 0.0}

    while not frontier.empty:
        current_node: Node[T] = frontier.pop()
        current_state: T = current_node.state

        if goal_test(current_state):
            print("Nodes visited:", len(explored))
            return current_node

        for child in successors(current_state):
            new_cost: float = current_node.cost + 1
            if child not in explored or explored[child] > new_cost:
                explored[child] = new_cost
                frontier.push(Node(child, current_node, new_cost, heuristic(child)))
    print("Nodes visited:", len(explored))
    return None



def limitedDepthFirstSearch(initial: T, goal_test: Callable[[T], bool], successors: Callable[[T], List[T]], DepthLimit) -> Optional[Node[T]]:
    frontier: List[T] = [Node(initial, None)]
    explored: Set[T] = set()
    DEPTH_LIMIT = DepthLimit

    while frontier:
        current_node: Node[T] = frontier.pop()
        current_state: T = current_node.state

        if goal_test(current_state):
            print("Nodes visited:", len(explored))
            return current_node

        # Switch to BFS if depth limit reached
        if current_node.cost >= DEPTH_LIMIT:
            frontier = deque(frontier)  # Convert to deque for efficient popping from the left
            while frontier:
                current_node = frontier.popleft()
                current_state = current_node.state

                if goal_test(current_state):
                    print("Nodes visited:", len(explored))
                    return current_node
                explored.add(current_state)

                for child in successors(current_state):
                    if child not in explored:
                        frontier.append(Node(child, current_node, current_node.cost + 1))
        else:
            for child in successors(current_state):
                if child not in explored:
                    explored.add(child)
                    frontier.append(Node(child, current_node, current_node.cost + 1))
    print("Nodes visited:", len(explored))
    return None

def astar_bidirectional(initial: T, goal: T, goal_test: Callable[[T], bool], successors: Callable[[T], List[T]],
                        heuristic: Callable[[T], float]) -> Optional[List[Node[T]]]:
    forward_frontier: PriorityQueue[Node[T]] = PriorityQueue()
    forward_frontier.push(Node(initial, None, 0.0, heuristic(initial), None))
    forward_explored: Dict[T, float] = {initial: 0.0}

    backward_frontier: PriorityQueue[Node[T]] = PriorityQueue()
    backward_frontier.push(Node(goal, None, 0.0, heuristic(goal), None))
    backward_explored: Dict[T, float] = {goal: 0.0}

    common_state = None

    while not forward_frontier.empty and not backward_frontier.empty:
        current_node = forward_frontier.pop()
        current_state = current_node.state

        current_node_back = backward_frontier.pop()
        current_state_back = current_node_back.state

        if goal_test(current_state):
            common_state = current_state
            break
        if current_state in backward_explored:
            common_state = current_state

            break

        if current_state_back in forward_explored:
            common_state = current_state
            break

        for child in successors(current_state):
            new_cost = current_node.cost + 1
            if child not in forward_explored or forward_explored[child] > new_cost:
                forward_explored[child] = new_cost
                forward_node = Node(child, current_node, new_cost, heuristic(child), current_node)
                forward_frontier.push(forward_node)

        for child in successors(current_state_back):  # Expand from current_state for the backward frontier
            new_cost = current_node_back.cost - 1
            if child not in backward_explored or backward_explored[child] < new_cost:
                backward_explored[child] = new_cost
                backward_node = Node(child, current_node_back, new_cost, heuristic(child), current_node_back)
                backward_frontier.push(backward_node)

    if common_state is None:
        print("Nodes visited:", len(forward_explored) + len(backward_explored))
        return None # Path not found

    # Construct the path from the common state
    forward_path = node_to_path(forward_frontier._container[0])  # Get the starting node from the forward frontier
    backward_path = node_to_path(backward_frontier._container[0])  # Get the starting node from the backward frontier
    backward_path.pop(0)  # Remove the common state from the backward path
    print("Nodes visited:", len(forward_explored) + len(backward_explored))
    return forward_path + backward_path[::-1]

#The below is an implementation of the limited A* search algorithm which i tried as research to see if it would be more efficient in finding the shortest path in a relatively short time period.
def limitedAstarSearch(initial: T, goal_test: Callable[[T], bool], successors: Callable[[T], List[T]], heuristic: Callable[[T], float], DepthLimit) -> Optional[Node[T]]:
    frontier: PriorityQueue[Node[T]] = PriorityQueueWithFunctions()
    frontier.push(Node(initial, None, 0.0, heuristic(initial), None), 0.0)
    explored: Set[T] = set()
    DEPTH_LIMIT = DepthLimit

    while frontier:
        current_node: Node[T] = frontier.pop()
        current_state: T = current_node.state

        if goal_test(current_state):
            print("Nodes visited:", len(explored))
            return current_node

        # Switch to A* if depth limit reached
        if current_node.cost >= DEPTH_LIMIT:
            while not frontier.empty():
                current_node = frontier.pop()
                current_state = current_node.state

                if goal_test(current_state):
                    print("Nodes visited:", len(explored))
                    return current_node
                explored.add(current_state)

                for child in successors(current_state):
                    if child not in explored:
                        explored.add(child)
                        frontier.push(Node(child, current_node, current_node.cost + 1, heuristic(child)), current_node.cost + 1 + heuristic(child))
        else:
            for child in successors(current_state):
                if child not in explored:
                    explored.add(child)
                    frontier.push(Node(child, current_node, current_node.cost + 1, heuristic(child)), current_node.cost + 1 + heuristic(child))
    print("Nodes visited:", len(explored))
    return None