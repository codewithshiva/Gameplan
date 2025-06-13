# Day 1-2 Complete Walkthrough: Classes, Objects, and Methods

## 🎯 Learning Goals
By the end of these 2 days, you'll understand:
- Why DSA problems use the `Solution` class pattern
- How this connects to your Django service classes
- How to implement and think about Two Sum problem
- The foundation for all future DSA problem-solving

---

## 📅 Day 1: Understanding the DSA Class Pattern

### Part 1: Theory (25 minutes)

#### 1.1 The Mystery of the Solution Class (10 minutes)

**You've probably wondered:** "Why do LeetCode problems always use this weird `Solution` class?"

```python
# This pattern appears EVERYWHERE in DSA
class Solution:
    def two_sum(self, nums, target):
        # Why not just a function?
        pass
```

**The Answer:** It's preparing you for real-world software design! Let me show you:

**DSA Context:**
```python
class Solution:
    def __init__(self):
        # Instance variables for the problem context
        self.memo = {}           # For caching/memoization
        self.operations = 0      # For tracking complexity
        self.debug_mode = False  # For testing
    
    def two_sum(self, nums, target):
        self.operations += 1
        if self.debug_mode:
            print(f"Processing array: {nums}")
        
        # Your algorithm here
        seen = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i
        return []
```

**Django Parallel (You already know this!):**
```python
class UserService:
    def __init__(self):
        self.cache = {}           # Same caching pattern!
        self.api_calls = 0        # Same tracking pattern!
        self.debug = settings.DEBUG
    
    def get_user_posts(self, user_id):
        self.api_calls += 1
        if self.debug:
            print(f"Fetching posts for user: {user_id}")
        
        # Your business logic here
        if user_id in self.cache:
            return self.cache[user_id]
        
        posts = Post.objects.filter(user_id=user_id)
        self.cache[user_id] = posts
        return posts
```

**💡 Aha Moment:** Both patterns solve the same problems - state management, caching, and reusability!

#### 1.2 Why Classes Over Functions? (10 minutes)

**Function Approach (Limited):**
```python
def two_sum(nums, target):
    # No memory between calls
    # No state tracking
    # Hard to extend
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Every call starts fresh - no optimization possible
result1 = two_sum([2, 7, 11, 15], 9)
result2 = two_sum([2, 7, 11, 15], 13)  # Recalculates everything!
```

**Class Approach (Powerful):**
```python
class TwoSumSolver:
    def __init__(self):
        self.cache = {}  # Remember previous results
        self.processed_arrays = {}
    
    def solve(self, nums, target):
        # Convert list to tuple for hashing
        nums_key = tuple(nums)
        
        # Check if we've seen this array before
        if nums_key not in self.processed_arrays:
            # Build the lookup table once
            lookup = {}
            for i, num in enumerate(nums):
                lookup[num] = i
            self.processed_arrays[nums_key] = lookup
        
        # Now solve using cached lookup
        lookup = self.processed_arrays[nums_key]
        for i, num in enumerate(nums):
            complement = target - num
            if complement in lookup and lookup[complement] != i:
                return [i, lookup[complement]]
        return []

# Usage - benefits from caching!
solver = TwoSumSolver()
result1 = solver.solve([2, 7, 11, 15], 9)   # Builds cache
result2 = solver.solve([2, 7, 11, 15], 13)  # Uses cache!
```

#### 1.3 Django Connection Deep Dive (5 minutes)

You already use this pattern in Django! Look at these examples:

**Django Model Manager (Class-based):**
```python
class ActiveUserManager(models.Manager):
    def __init__(self):
        super().__init__()
        self.cache_timeout = 300
        
    def active_users(self):
        # Same state management as DSA!
        return self.filter(is_active=True)
```

**Django View Classes:**
```python
class UserListView(APIView):
    def __init__(self):
        super().__init__()
        self.pagination_size = 20  # Instance state!
    
    def get(self, request):
        # Method operates on instance state
        users = User.objects.all()[:self.pagination_size]
        return Response(users)
```

---

### Part 2: Implementation (25 minutes)

#### 2.1 Your First DSA Class (15 minutes)

Let's implement the Two Sum problem step by step:

```python
class Solution:
    def __init__(self):
        """
        Initialize the solution instance
        Think of this like Django's __init__ in views/services
        """
        self.memo = {}  # We'll use this for future problems
        self.call_count = 0  # Track how many times we solve problems
    
    def two_sum(self, nums, target):
        """
        Find two numbers that add up to target
        Args:
            nums: List[int] - array of integers
            target: int - target sum
        Returns:
            List[int] - indices of the two numbers
        """
        self.call_count += 1
        
        # The algorithm - hash map approach
        seen = {}  # number -> index mapping
        
        for i, num in enumerate(nums):
            complement = target - num
            
            if complement in seen:
                # Found the pair!
                return [seen[complement], i]
            
            # Store current number and its index
            seen[num] = i
        
        # No solution found
        return []
    
    def get_stats(self):
        """Bonus method - track usage like in Django services"""
        return f"Solved {self.call_count} problems"

# Test your implementation
solution = Solution()

# Test Case 1
nums1 = [2, 7, 11, 15]
target1 = 9
result1 = solution.two_sum(nums1, target1)
print(f"Input: {nums1}, Target: {target1}")
print(f"Output: {result1}")  # Should be [0, 1]

# Test Case 2
nums2 = [3, 2, 4]
target2 = 6
result2 = solution.two_sum(nums2, target2)
print(f"Input: {nums2}, Target: {target2}")
print(f"Output: {result2}")  # Should be [1, 2]

print(solution.get_stats())  # Shows call count
```

#### 2.2 Understanding the Algorithm (10 minutes)

**Step-by-step walkthrough of Two Sum:**

```python
# Let's trace through nums = [2, 7, 11, 15], target = 9

# Iteration 1: i=0, num=2
seen = {}
complement = 9 - 2 = 7
# 7 not in seen, so add: seen = {2: 0}

# Iteration 2: i=1, num=7  
seen = {2: 0}
complement = 9 - 7 = 2
# 2 IS in seen! seen[2] = 0
# Return [0, 1] ✅
```

**Why this works:**
- We're looking for two numbers: `a + b = target`
- When we see number `a`, we look for `target - a`
- Hash map gives us O(1) lookup time
- One pass through the array = O(n) time complexity

---

### Part 3: Django Connection (10 minutes)

#### 3.1 Same Pattern in Your Django Code

```python
# This is probably in your Django project somewhere:
class UserAuthService:
    def __init__(self):
        self.failed_attempts = {}  # Like our 'seen' dictionary!
        self.max_attempts = 3
    
    def authenticate_user(self, username, password):
        # Same pattern: check existence, then decide
        if username in self.failed_attempts:
            if self.failed_attempts[username] >= self.max_attempts:
                return {"error": "Account locked"}
        
        # Authenticate logic here
        if self.is_valid_credentials(username, password):
            # Reset on success
            self.failed_attempts.pop(username, None)
            return {"success": True}
        else:
            # Track failure
            self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
            return {"error": "Invalid credentials"}
```

**See the similarity?**
- Hash map for fast lookups ✓
- State management ✓  
- Instance methods ✓
- Return meaningful results ✓

---

## 📅 Day 2: Valid Parentheses - Your First Stack Class

### Part 1: Theory (25 minutes)

#### 2.1 Why Stacks? (10 minutes)

**Real-world Stack Examples you know:**
- **Django Middleware Stack:** Each middleware wraps the next
- **Function Call Stack:** Python manages function calls
- **Browser History:** Back button uses stack behavior
- **Undo Operations:** Last action undone first

**Stack Behavior:**
```python
# Last In, First Out (LIFO)
stack = []
stack.append("first")   # Push
stack.append("second")  # Push  
stack.append("third")   # Push

print(stack.pop())      # "third" - last in, first out
print(stack.pop())      # "second"
print(stack.pop())      # "first"
```

#### 2.2 Valid Parentheses Problem (10 minutes)

**Problem:** Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

**Examples:**
- `"()"` → True
- `"()[]{}"` → True  
- `"(]"` → False
- `"([)]"` → False
- `"{[]}"` → True

**Why Stack Works:**
- When we see opening bracket: push to stack
- When we see closing bracket: check if it matches top of stack
- Stack empty at end = valid string

#### 2.3 Design Your Stack Class (5 minutes)

```python
class ValidParenthesesChecker:
    def __init__(self):
        self.stack = []
        self.brackets = {
            ')': '(',
            '}': '{', 
            ']': '['
        }
        self.check_count = 0
    
    def is_valid(self, s):
        # We'll implement this next
        pass
```

---

### Part 2: Implementation (25 minutes)

#### 2.1 Build the Stack-based Solution (20 minutes)

```python
class ValidParenthesesChecker:
    def __init__(self):
        """Initialize the parentheses checker"""
        self.stack = []
        self.brackets_map = {
            ')': '(',
            '}': '{',
            ']': '['
        }
        self.validation_count = 0
    
    def is_valid(self, s):
        """
        Check if parentheses string is valid
        Args:
            s: str - string containing brackets
        Returns:
            bool - True if valid, False otherwise
        """
        self.validation_count += 1
        self.stack = []  # Reset stack for each validation
        
        for char in s:
            if char in self.brackets_map:
                # Closing bracket found
                if not self.stack:
                    # No opening bracket to match
                    return False
                
                top_element = self.stack.pop()
                if self.brackets_map[char] != top_element:
                    # Mismatched brackets
                    return False
            else:
                # Opening bracket - push to stack
                self.stack.append(char)
        
        # Valid if stack is empty (all brackets matched)
        return len(self.stack) == 0
    
    def get_stats(self):
        """Get usage statistics"""
        return f"Validated {self.validation_count} strings"

# Test your implementation
checker = ValidParenthesesChecker()

test_cases = [
    "()",           # True
    "()[]{}", 	    # True
    "(]",           # False
    "([)]",         # False
    "{[]}",         # True
    "(((",          # False
    ")))",          # False
]

for test in test_cases:
    result = checker.is_valid(test)
    print(f"'{test}' -> {result}")

print(checker.get_stats())
```

#### 2.2 Trace Through an Example (5 minutes)

**Let's trace `"{[]}"`:**

```python
# Input: "{[]}"
stack = []

# char = '{'
# '{' not in brackets_map, so it's opening
stack = ['{']

# char = '['  
# '[' not in brackets_map, so it's opening
stack = ['{', '[']

# char = ']'
# ']' in brackets_map, brackets_map[']'] = '['
# stack.pop() = '[', matches brackets_map[']']
stack = ['{']

# char = '}'
# '}' in brackets_map, brackets_map['}'] = '{'  
# stack.pop() = '{', matches brackets_map['}']
stack = []

# End: stack is empty -> return True ✅
```

---

### Part 3: Django Connection (10 minutes)

#### 3.1 Stack Pattern in Django

```python
# Django Middleware - Stack behavior!
class SecurityMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.request_stack = []  # Track request processing
    
    def __call__(self, request):
        # Push request context (like opening bracket)
        self.request_stack.append({
            'timestamp': timezone.now(),
            'path': request.path
        })
        
        response = self.get_response(request)
        
        # Pop request context (like closing bracket)
        request_context = self.request_stack.pop()
        
        # Add processing time to response
        processing_time = timezone.now() - request_context['timestamp']
        response['X-Processing-Time'] = str(processing_time.total_seconds())
        
        return response
```

**Same Pattern:**
- Push on entry ✓
- Pop on exit ✓  
- Validate matching ✓
- State management ✓

---

## 🎯 End of Day 1-2 Assessment

### Quick Self-Check:
1. ✅ Can you explain why DSA uses classes instead of functions?
2. ✅ Do you understand the Two Sum algorithm?
3. ✅ Can you trace through the Valid Parentheses algorithm?
4. ✅ Do you see the connection to Django patterns?

### Your Homework for Tomorrow (Day 3):
1. **Solve one more Two Sum variant:** "Two Sum II - Input Array Is Sorted"
2. **Implement:** A simple calculator using the stack (bonus challenge)
3. **Think:** What other Django patterns use stack-like behavior?

### Problems to Practice:
- **Easy:** Two Sum, Valid Parentheses, Remove Duplicates from Sorted Array
- **Try:** Implement these using class-based solutions
- **Connect:** Think about how each pattern appears in your Django work

---
