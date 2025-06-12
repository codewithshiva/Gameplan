# Complete DSA + OOP Integrated Roadmap for Backend Developers

## Phase 0: Foundation with OOP Integration (3-4 weeks)

### Week 1: Python OOP Fundamentals + Basic DSA
**Goal:** Understand why DSA problems use classes and connect to your Django experience

**Daily Structure (1 hour):**
- **25 min:** OOP concept + implementation
- **25 min:** Apply to simple DSA problem
- **10 min:** Connect to Django patterns

**Day 1-2: Classes, Objects, and Methods**
```python
# DSA Context
class Solution:
    def __init__(self):
        self.memo = {}  # For memoization later
    
    def two_sum(self, nums, target):
        # Your algorithm here
        pass

# Django Connection
class UserService:
    def __init__(self):
        self.cache = {}
    
    def get_user_posts(self, user_id):
        # Similar pattern to DSA solutions
        pass
```

**Day 3-4: Instance vs Class Variables**
```python
# DSA: Counter class for frequency problems
class Counter:
    def __init__(self):
        self.count = {}  # Instance variable
    
    def add(self, item):
        self.count[item] = self.count.get(item, 0) + 1

# Django: Model with class variables
class User(models.Model):
    STATUS_CHOICES = [...]  # Class variable
    status = models.CharField(...)  # Instance field
```

**Day 5-7: Constructor Patterns and Method Chaining**
```python
# DSA: LinkedList implementation
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        # Implementation
        return self  # Enable chaining

# Django: QuerySet chaining pattern
# User.objects.filter(...).exclude(...).order_by(...)
```

**Problems to Solve (Week 1):**
- Two Sum (using Solution class structure)
- Valid Parentheses (implement Stack class)
- Reverse String (using custom String class)

### Week 2: Data Structures as Classes
**Goal:** Build core data structures with proper OOP design

**Stack Implementation with OOP Best Practices:**
```python
class Stack:
    def __init__(self):
        self._items = []  # Private attribute
    
    def push(self, item):
        self._items.append(item)
        return self  # Method chaining
    
    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._items.pop()
    
    def peek(self):
        if self.is_empty():
            return None
        return self._items[-1]
    
    def is_empty(self):
        return len(self._items) == 0
    
    def __len__(self):  # Magic method
        return len(self._items)
    
    def __str__(self):  # String representation
        return f"Stack({self._items})"
    
    def __repr__(self):
        return self.__str__()

# Django Connection: Custom Manager
class ActiveUserManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(is_active=True)
    
    def with_posts(self):
        return self.get_queryset().prefetch_related('posts')
```

**Queue with Inheritance:**
```python
from collections import deque

class Queue:
    def __init__(self):
        self._items = deque()
    
    def enqueue(self, item):
        self._items.append(item)
        return self
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items.popleft()
    
    def is_empty(self):
        return len(self._items) == 0

# Priority Queue inherits from Queue
class PriorityQueue(Queue):
    def __init__(self):
        import heapq
        self._items = []
    
    def enqueue(self, item, priority):
        heapq.heappush(self._items, (priority, item))
        return self
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        return heapq.heappop(self._items)[1]
```

**Problems to Solve (Week 2):**
- Baseball Game (using your Stack class)
- Recent Counter (using your Queue class)
- Valid Parentheses (enhanced with custom Stack)

### Week 3: Advanced OOP Patterns in DSA
**Goal:** Learn design patterns commonly used in both DSA and Django

**Factory Pattern for Data Structures:**
```python
class DataStructureFactory:
    @staticmethod
    def create_stack():
        return Stack()
    
    @staticmethod
    def create_queue():
        return Queue()
    
    @classmethod
    def create_by_type(cls, ds_type):
        if ds_type == 'stack':
            return cls.create_stack()
        elif ds_type == 'queue':
            return cls.create_queue()
        else:
            raise ValueError(f"Unknown type: {ds_type}")

# Django Connection: Factory pattern in views
class ViewFactory:
    @staticmethod
    def create_list_view(model):
        class ListView(generics.ListAPIView):
            queryset = model.objects.all()
        return ListView
```

**Strategy Pattern for Algorithms:**
```python
from abc import ABC, abstractmethod

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data):
        pass

class BubbleSort(SortStrategy):
    def sort(self, data):
        # Implementation
        pass

class QuickSort(SortStrategy):
    def sort(self, data):
        # Implementation
        pass

class Sorter:
    def __init__(self, strategy: SortStrategy):
        self.strategy = strategy
    
    def sort_data(self, data):
        return self.strategy.sort(data)

# Django Connection: Strategy for different authentication methods
class AuthStrategy(ABC):
    @abstractmethod
    def authenticate(self, credentials):
        pass
```

**Problems to Solve (Week 3):**
- Implement Binary Search Tree class
- Design HashSet class
- Create custom Array class with sorting strategies

### Week 4: OOP + Recursion Integration
**Goal:** Understand how OOP and recursion work together

**Tree Node with OOP:**
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def insert(self, val):
        """Insert value maintaining BST property"""
        if val < self.val:
            if self.left is None:
                self.left = TreeNode(val)
            else:
                self.left.insert(val)
        else:
            if self.right is None:
                self.right = TreeNode(val)
            else:
                self.right.insert(val)
    
    def search(self, val):
        """Search for value in BST"""
        if self.val == val:
            return True
        elif val < self.val and self.left:
            return self.left.search(val)
        elif val > self.val and self.right:
            return self.right.search(val)
        return False
    
    def inorder_traversal(self):
        """Return inorder traversal as list"""
        result = []
        if self.left:
            result.extend(self.left.inorder_traversal())
        result.append(self.val)
        if self.right:
            result.extend(self.right.inorder_traversal())
        return result
```

**Problems to Solve (Week 4):**
- Maximum Depth of Binary Tree (using TreeNode class)
- Validate Binary Search Tree
- Path Sum problems

## Phase 1: Core Patterns with OOP (4-5 weeks)

### Week 5-6: Two Pointers + OOP Design
**Goal:** Implement two-pointer techniques with clean class design

**Array Processor Class:**
```python
class ArrayProcessor:
    def __init__(self, arr):
        self.arr = arr
        self.left = 0
        self.right = len(arr) - 1
    
    def two_sum_sorted(self, target):
        """Two pointers for sorted array"""
        while self.left < self.right:
            current_sum = self.arr[self.left] + self.arr[self.right]
            if current_sum == target:
                return [self.left, self.right]
            elif current_sum < target:
                self.left += 1
            else:
                self.right -= 1
        return []
    
    def is_palindrome(self):
        """Check if array is palindrome"""
        left, right = 0, len(self.arr) - 1
        while left < right:
            if self.arr[left] != self.arr[right]:
                return False
            left += 1
            right -= 1
        return True
    
    def reverse_in_place(self):
        """Reverse array using two pointers"""
        left, right = 0, len(self.arr) - 1
        while left < right:
            self.arr[left], self.arr[right] = self.arr[right], self.arr[left]
            left += 1
            right -= 1
        return self

# Django Connection: Custom QuerySet methods
class CustomQuerySet(models.QuerySet):
    def with_stats(self):
        return self.annotate(
            post_count=models.Count('posts'),
            avg_rating=models.Avg('posts__rating')
        )
```

**Problems to Solve (Week 5-6):**
- Container With Most Water
- 3Sum (implement ThreeSum class)
- Remove Duplicates from Sorted Array
- Valid Palindrome

### Week 7-8: Sliding Window + State Management
**Goal:** Master sliding window with proper state management using OOP

**Sliding Window Manager:**
```python
class SlidingWindowManager:
    def __init__(self, data):
        self.data = data
        self.window_start = 0
        self.current_sum = 0
        self.char_frequency = {}
    
    def max_sum_subarray(self, k):
        """Maximum sum subarray of size k"""
        if len(self.data) < k:
            return 0
        
        # Initial window
        for i in range(k):
            self.current_sum += self.data[i]
        
        max_sum = self.current_sum
        
        # Slide the window
        for i in range(k, len(self.data)):
            self.current_sum = self.current_sum - self.data[i-k] + self.data[i]
            max_sum = max(max_sum, self.current_sum)
        
        return max_sum
    
    def longest_substring_k_distinct(self, k):
        """Longest substring with at most k distinct characters"""
        if k == 0:
            return 0
        
        max_length = 0
        
        for window_end in range(len(self.data)):
            # Expand window
            right_char = self.data[window_end]
            self.char_frequency[right_char] = self.char_frequency.get(right_char, 0) + 1
            
            # Contract window if needed
            while len(self.char_frequency) > k:
                left_char = self.data[self.window_start]
                self.char_frequency[left_char] -= 1
                if self.char_frequency[left_char] == 0:
                    del self.char_frequency[left_char]
                self.window_start += 1
            
            max_length = max(max_length, window_end - self.window_start + 1)
        
        return max_length

# Django Connection: Pagination with state
class StatefulPaginator:
    def __init__(self, queryset, page_size):
        self.queryset = queryset
        self.page_size = page_size
        self.current_page = 1
        self.cache = {}
    
    def get_page(self, page_num):
        if page_num in self.cache:
            return self.cache[page_num]
        
        offset = (page_num - 1) * self.page_size
        page_data = self.queryset[offset:offset + self.page_size]
        self.cache[page_num] = page_data
        return page_data
```

**Problems to Solve (Week 7-8):**
- Longest Substring Without Repeating Characters
- Minimum Window Substring
- Sliding Window Maximum
- Fruits into Baskets

### Week 9-10: Trees and Graphs with OOP
**Goal:** Advanced tree/graph algorithms with proper OOP design

**Enhanced Tree Class:**
```python
class BinaryTree:
    def __init__(self, root_val=None):
        self.root = TreeNode(root_val) if root_val is not None else None
        self.size = 1 if root_val is not None else 0
    
    def insert_level_order(self, values):
        """Insert values in level order"""
        if not values:
            return
        
        self.root = TreeNode(values[0])
        queue = [self.root]
        i = 1
        
        while queue and i < len(values):
            node = queue.pop(0)
            
            if i < len(values) and values[i] is not None:
                node.left = TreeNode(values[i])
                queue.append(node.left)
            i += 1
            
            if i < len(values) and values[i] is not None:
                node.right = TreeNode(values[i])
                queue.append(node.right)
            i += 1
        
        self.size = len([v for v in values if v is not None])
    
    def traverse(self, method='inorder'):
        """Flexible traversal method"""
        if not self.root:
            return []
        
        traversal_methods = {
            'inorder': self._inorder,
            'preorder': self._preorder,
            'postorder': self._postorder,
            'levelorder': self._level_order
        }
        
        return traversal_methods.get(method, self._inorder)(self.root)
    
    def _inorder(self, node):
        if not node:
            return []
        return self._inorder(node.left) + [node.val] + self._inorder(node.right)

# Graph class with adjacency list
class Graph:
    def __init__(self, directed=False):
        self.graph = {}
        self.directed = directed
    
    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []
    
    def add_edge(self, u, v):
        self.add_vertex(u)
        self.add_vertex(v)
        self.graph[u].append(v)
        if not self.directed:
            self.graph[v].append(u)
    
    def dfs(self, start_vertex, visited=None):
        if visited is None:
            visited = set()
        
        visited.add(start_vertex)
        result = [start_vertex]
        
        for neighbor in self.graph.get(start_vertex, []):
            if neighbor not in visited:
                result.extend(self.dfs(neighbor, visited))
        
        return result
    
    def bfs(self, start_vertex):
        visited = set()
        queue = [start_vertex]
        result = []
        
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                queue.extend([v for v in self.graph.get(vertex, []) if v not in visited])
        
        return result

# Django Connection: Custom model methods
class Post(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def get_related_posts(self, limit=5):
        """Get related posts using graph-like relationship"""
        # Similar to graph traversal
        pass
    
    @classmethod
    def build_recommendation_graph(cls):
        """Build graph of post relationships"""
        # Similar to our Graph class
        pass
```

**Problems to Solve (Week 9-10):**
- Binary Tree Level Order Traversal
- Validate Binary Search Tree
- Number of Islands (using Graph class)
- Course Schedule (topological sort)

## Phase 2: Advanced Patterns with Design Patterns (5-6 weeks)

### Week 11-12: Dynamic Programming with Memoization Classes
**Goal:** DP with proper OOP design and memoization patterns

**Memoization Decorator and DP Solver:**
```python
from functools import wraps

class Memoizer:
    def __init__(self):
        self.cache = {}
    
    def memoize(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            if key not in self.cache:
                self.cache[key] = func(*args, **kwargs)
            return self.cache[key]
        return wrapper
    
    def clear_cache(self):
        self.cache.clear()

class DPSolver:
    def __init__(self):
        self.memo = Memoizer()
    
    @property
    def fibonacci(self):
        @self.memo.memoize
        def fib(n):
            if n <= 1:
                return n
            return fib(n-1) + fib(n-2)
        return fib
    
    @property
    def coin_change(self):
        @self.memo.memoize
        def change(amount, coins):
            if amount == 0:
                return 0
            if amount < 0:
                return float('inf')
            
            min_coins = float('inf')
            for coin in coins:
                result = change(amount - coin, coins)
                if result != float('inf'):
                    min_coins = min(min_coins, result + 1)
            
            return min_coins
        return change

# Django Connection: Caching strategies
class CacheManager:
    def __init__(self):
        self.cache = {}
        self.ttl = {}
    
    def get_or_compute(self, key, compute_func, ttl=300):
        import time
        current_time = time.time()
        
        if key in self.cache and key in self.ttl:
            if current_time - self.ttl[key] < ttl:
                return self.cache[key]
        
        result = compute_func()
        self.cache[key] = result
        self.ttl[key] = current_time
        return result
```

### Week 13-14: Backtracking with State Management
**Goal:** Backtracking algorithms with proper state management

**Backtracking Framework:**
```python
class BacktrackingSolver:
    def __init__(self):
        self.solutions = []
        self.current_path = []
    
    def solve_n_queens(self, n):
        self.board = [[False] * n for _ in range(n)]
        self.solutions = []
        self._n_queens_helper(0, n)
        return self.solutions
    
    def _n_queens_helper(self, row, n):
        if row == n:
            # Found a valid solution
            solution = []
            for r in range(n):
                row_str = ""
                for c in range(n):
                    row_str += "Q" if self.board[r][c] else "."
                solution.append(row_str)
            self.solutions.append(solution)
            return
        
        for col in range(n):
            if self._is_safe(row, col, n):
                # Make choice
                self.board[row][col] = True
                # Recurse
                self._n_queens_helper(row + 1, n)
                # Backtrack
                self.board[row][col] = False
    
    def _is_safe(self, row, col, n):
        # Check column
        for i in range(row):
            if self.board[i][col]:
                return False
        
        # Check diagonals
        for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
            if self.board[i][j]:
                return False
        
        for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
            if self.board[i][j]:
                return False
        
        return True

# Django Connection: Form validation with backtracking
class FormValidator:
    def __init__(self):
        self.errors = []
        self.valid_states = []
    
    def validate_complex_form(self, form_data, rules):
        # Similar backtracking approach for complex validation
        pass
```

### Week 15-16: System Design Patterns
**Goal:** Apply DSA + OOP to system design problems

**LRU Cache Implementation:**
```python
class LRUNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> node
        
        # Create dummy head and tail
        self.head = LRUNode()
        self.tail = LRUNode()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node):
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """Remove an existing node"""
        prev_node = node.prev
        new_node = node.next
        
        prev_node.next = new_node
        new_node.prev = prev_node
    
    def _move_to_head(self, node):
        """Move node to head"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self):
        """Pop the last node"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key: int) -> int:
        node = self.cache.get(key)
        
        if node:
            # Move to head
            self._move_to_head(node)
            return node.value
        
        return -1
    
    def put(self, key: int, value: int) -> None:
        node = self.cache.get(key)
        
        if not node:
            new_node = LRUNode(key, value)
            
            if len(self.cache) >= self.capacity:
                # Remove tail
                tail = self._pop_tail()
                del self.cache[tail.key]
            
            self.cache[key] = new_node
            self._add_node(new_node)
        else:
            # Update value and move to head
            node.value = value
            self._move_to_head(node)

# Django Connection: Database connection pool
class ConnectionPool:
    def __init__(self, max_connections=10):
        self.max_connections = max_connections
        self.available_connections = []
        self.active_connections = set()
    
    def get_connection(self):
        # Similar to LRU cache management
        pass
    
    def release_connection(self, connection):
        # Return connection to pool
        pass
```

## Integration with Django Throughout

**Weekly Django Connections:**
- **Week 1-2:** Model design, Manager patterns
- **Week 3-4:** Custom QuerySet methods, Factory patterns  
- **Week 5-6:** View classes, Method chaining
- **Week 7-8:** Pagination, Caching strategies
- **Week 9-10:** Recommendation systems, Graph relationships
- **Week 11-12:** Database query optimization, Memoization
- **Week 13-14:** Form validation, State management
- **Week 15-16:** System architecture, Connection pooling

**Final Project Ideas:**
1. **Social Media Backend:** Friend recommendation using graph algorithms
2. **E-commerce Search:** Product search with trie data structure
3. **Real-time Chat:** Message ordering with heap/priority queue
4. **Content Management:** LRU cache for frequently accessed content
