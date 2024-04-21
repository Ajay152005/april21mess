def threeSum(nums):
    nums.sort()
    result = []
    n = len(nums)

    for i in range(n):
        # Avoid duplicates
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, n -1

        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                # Avoid duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right -1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result

# TEST CASES

print(threeSum([-1,0,1,2,-1,-4])) # Output: [[-1, -1, 2], [-1, 0, 1]]
print(threeSum([0,1,1]))   # Output: []
print(threeSum([0,0,0]))  #Output: [[0,0,0]]