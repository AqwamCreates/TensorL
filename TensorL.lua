local TensorL = {}

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else -- number, string, boolean, etc

		copy = original

	end

	return copy

end

local function getNumberOfDimensions(tensor)
	
	if (typeof(tensor) ~= "table") then return 0 end
	
	return getNumberOfDimensions(tensor[1]) + 1
	
end

local function getSubTensorLength(tensor, targetDimension)
	
	local numberOfDimensions = getNumberOfDimensions(tensor)
	
	if (numberOfDimensions == targetDimension) then return #tensor end
	
	return getSubTensorLength(tensor[1], targetDimension)
	
end

local function getDimensionArray(tensor)
	
	local numberOfDimensions = getNumberOfDimensions(tensor)

	local dimensionArray = {}

	for dimension = numberOfDimensions, 1, -1  do

		local length = getSubTensorLength(tensor, dimension)

		table.insert(dimensionArray, length)

	end
	
	return dimensionArray
	
end

local function applyOperation(operation, tensor1, tensor2)
	
	local dimensionArray1 = getDimensionArray(tensor1)
	
	local dimensionArray2 = getDimensionArray(tensor2)
	
	for i, _ in ipairs(dimensionArray1) do if (dimensionArray1[i] ~= dimensionArray2[i]) then error("Invalid dimensions.") end end

	local result = {}
	
	if (#dimensionArray1 > 1) then
		
		for i = 1, #tensor1 do result[i] = applyOperation(operation, tensor1[i], tensor2[i]) end
		
	else
		
		for i = 1, #tensor1 do result[i] = operation(tensor1[i], tensor2[i]) end
		
	end
	
	return result

end

local function createString(tensor)

	local dimensionArray = getDimensionArray(tensor)
	
	local tensorLength = #tensor

	local result = " "

	if (#dimensionArray > 1) then
		
		result = result .. "{"

		for i = 1, #tensor do 
			
			result = result .. createString(tensor[i])
			
			if (i == tensorLength) then continue end
			
			result = result .. "\n"
			
		end
		
		result = result .. " }"

	else

		result = result .. "{ "

		for i = 1, tensorLength do 

			result = result .. tensor[i]

			if (i == tensorLength) then continue end

			result = result .. ", "

		end

		result = result .. " }"

	end

	return result

end

local function tensorProduct(tensor1, tensor2)
	
	local dimensionArray1 = getDimensionArray(tensor1)
	
	local dimensionArray2 = getDimensionArray(tensor2)

	for i, _ in ipairs(dimensionArray1) do if (dimensionArray1[i] ~= dimensionArray2[i]) then error("Invalid dimensions.") end end

	local numberOfValues = dimensionArray1[1]
	
	local result = {}

	if (#dimensionArray1 > 1) then
		
		for i = 1, numberOfValues, 1 do
			
			local subproduct = tensorProduct(tensor1[i], tensor2[i])
			
			table.insert(result, subproduct)
			
		end
		
	else
		
		for i = 1, numberOfValues, 1 do
			
			table.insert(result, tensor1[i] * tensor2[i])
			
		end
		
	end

	return result
end

local function innerProduct(tensor1, tensor2)
	
	local dimensionArray1 = getDimensionArray(tensor1)

	local dimensionArray2 = getDimensionArray(tensor2)

	for i, _ in ipairs(dimensionArray1) do if (dimensionArray1[i] ~= dimensionArray2[i]) then error("Invalid dimensions.") end end
	
	local numberOfValues = dimensionArray1[1]
	
	local result = 0

	if (#dimensionArray1 > 1) then
		
		for i = 1, numberOfValues, 1 do result += innerProduct(tensor1[i], tensor2[i]) end
		
	else
		
		for i = 1, numberOfValues, 1 do result += (tensor1[i] * tensor2[i]) end
		
	end

	return result
	
end

local function outerProduct(tensor1, tensor2)
	
	local dimensionArray1 = getDimensionArray(tensor1)
	
	local dimensionArray2 = getDimensionArray(tensor2)

	for i, _ in ipairs(dimensionArray1) do if dimensionArray1[i] ~= dimensionArray2[i] then error("Invalid dimensions.") end end

	local numberOfValues = dimensionArray1[1]
	
	local result = {}

	if (#dimensionArray1 > 1) then
		
		for i = 1, numberOfValues do result[i] = outerProduct(tensor1[i], tensor2[i]) end
		
	else
		
		for i = 1, numberOfValues do
			
			result[i] = {}
			
			for j = 1, numberOfValues do result[i][j] = tensor1[i] * tensor2[j] end
			
		end
		
	end

	return result
	
end

local function eq(booleanTensor)
	
	local dimensionArray1 = getDimensionArray(booleanTensor)

	local numberOfValues = dimensionArray1[1]

	local result = true

	if (#dimensionArray1 > 1) then

		for i = 1, numberOfValues do result = eq(booleanTensor[i]) end

	else

		for i = 1, numberOfValues do 
			
			result = (result == booleanTensor[i])
			
			if (result == false) then return false end
			
		end

	end

	return result
	
end

local function createTensor(dimensionArray, initialValue)
	
	local result = {}
	
	if (#dimensionArray > 2) then
		
		local remainingDimensions = {}
		
		for i = 2, #dimensionArray do table.insert(remainingDimensions, dimensionArray[i]) end

		for i = 1, dimensionArray[1] do result[i] = createTensor(remainingDimensions, initialValue) end

	else
		
		for i = 1, dimensionArray[1] do result[i] = table.create(dimensionArray[2], initialValue) end
		
	end
	
	return result
	
end

function TensorL.new(...)
	
	local self = setmetatable({}, TensorL)

	self.Values = ...

	return self
	
end

function TensorL.create(dimensionArray, initialValue)
	
	initialValue = initialValue or 0
	
	local self = setmetatable({}, TensorL)
	
	self.Values = createTensor(dimensionArray, initialValue)
	
	return self
	
end

function TensorL:broadcast(dimensionsArray, values)

	local isNumber = typeof(values) == "number"

	if isNumber then return self.create(dimensionsArray, values) end
	
	return values

end

function TensorL:getNumberOfDimensions()

	return getNumberOfDimensions(self)

end

function TensorL:getDimensionArray()
	
	return getDimensionArray(self)
	
end

function TensorL:print()

	print(self)
	
end

function TensorL:transpose(dimension1, dimension2)
	
	if (typeof(dimension1) ~= "number") or (typeof(dimension2) ~= "number") then error("Dimensions are not numbers.") end
	
	local size = self:getSize()
	
	local result = {}

	-- Check if the specified dimensions are within the valid range
	if (dimension1 < 1) or (dimension1 > size[1]) or (dimension2 < 1) or (dimension2 > size[1]) or (dimension1 == dimension2) then
		error("Invalid dimensions for transpose.")
	end

	-- Initialize the transposed tensor with the same dimensions as the input tensor
	for i = 1, size[1] do
		
		result[i] = {}
		
		for j = 1, #self[i] do
			
			result[i][j] = {}
			
		end
		
	end

	-- Perform the transpose operation
	for i = 1, size[1] do
		
		for j = 1, #self[i] do
			
			for k = 1, #self[i][j] do
				
				result[i][j][k] = self[i][j][k]
				
			end
			
		end
		
	end

	-- Swap the specified dimensions
	for i = 1, size[1] do
		
		for j = 1, #self[i] do
			
			for k = 1, #self[i][j] do
				
				if dimension1 ~= i and dimension2 ~= i then
					
					result[i][j][k] = self[i][j][k]
					
				elseif dimension1 == i then
					
					result[dimension2][j][k] = self[i][j][k]
					
				elseif dimension2 == i then
					
					result[dimension1][j][k] = self[i][j][k]
					
				end
				
			end
			
		end
		
	end

	return self.new(result)
	
end

function TensorL:__eq(other)
	
	local numberOfDimensions1 = getNumberOfDimensions(self)

	local numberOfDimensions2 = getNumberOfDimensions(other)

	if (numberOfDimensions1 ~= numberOfDimensions2) then return false end
	
	local operation = function(a, b) return (a == b) end

	local result = applyOperation(operation, self, other)
	
	local isEqual = eq(result)

	return isEqual
	
end

function TensorL:isEqualTo(other)

	local numberOfDimensions1 = getNumberOfDimensions(self)

	local numberOfDimensions2 = getNumberOfDimensions(other)
	
	if (numberOfDimensions1 ~= numberOfDimensions2) then error("Invalid dimensions.") end

	local operation = function(a, b) return (a == b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)

end

function TensorL:isGreaterThan(other)

	local numberOfDimensions1 = getNumberOfDimensions(self)

	local numberOfDimensions2 = getNumberOfDimensions(other)

	if (numberOfDimensions1 ~= numberOfDimensions2) then error("Invalid dimensions.") end

	local operation = function(a, b) return (a > b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)

end

function TensorL:isGreaterOrEqualTo(other)

	local numberOfDimensions1 = getNumberOfDimensions(self)

	local numberOfDimensions2 = getNumberOfDimensions(other)

	if (numberOfDimensions1 ~= numberOfDimensions2) then error("Invalid dimensions.") end

	local operation = function(a, b) return (a >= b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)

end

function TensorL:isLessThan(other)

	local numberOfDimensions1 = getNumberOfDimensions(self)

	local numberOfDimensions2 = getNumberOfDimensions(other)

	if (numberOfDimensions1 ~= numberOfDimensions2) then error("Invalid dimensions.") end
	
	local operation = function(a, b) return (a < b) end

	local result = applyOperation(operation, self, other)
	
	return self.new(result)

end

function TensorL:isLessOrEqualTo(other)

	local numberOfDimensions1 = getNumberOfDimensions(self)

	local numberOfDimensions2 = getNumberOfDimensions(other)

	if (numberOfDimensions1 ~= numberOfDimensions2) then error("Invalid dimensions.") end

	local operation = function(a, b) return (a <= b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)

end

function TensorL:tensorProduct(other)
	
	local result = tensorProduct(self, other)
	
	return self.new(result)
	
end

function TensorL:innerProduct(other)

	return innerProduct(self, other)

end

function TensorL:outerProduct(other)
	
	local result = outerProduct(self, other)

	return self.create(result)

end

function TensorL:copy()
	
	return deepCopyTable(self)
	
end

function TensorL:__add(other)
	
	local numberOfDimensions = self:getNumberOfDimensions()
	
	local other = self:broadcast(numberOfDimensions, other)
	
	local operation = function(a, b) return (a + b) end
	
	local result = applyOperation(operation, self, other)

	return self.new(result)
	
end

function TensorL:__sub(other)
	
	local other = self:broadcast(other, table.unpack(self:getNumberOfDimensions()))

	local operation = function(a, b) return (a - b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)
	
end

function TensorL:__mul(other)
	
	local other = self:broadcast(other, table.unpack(self:getNumberOfDimensions()))

	local operation = function(a, b) return (a * b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)
	
end

function TensorL:__div(other)
	
	local other = self:broadcast(other, table.unpack(self:getNumberOfDimensions()))

	local operation = function(a, b) return (a / b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)
	
end

function TensorL:__unm(other)
	
	local other = self:broadcast(-1, table.unpack(self:getNumberOfDimensions()))

	local operation = function(a, b) return (a * b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)
	
end

function TensorL:__tostring()
	
	local text = "\n\n" .. createString(self) .. "\n\n"

	return text
	
end

function TensorL:__len()
	
	return #self.Values
	
end

function TensorL:__index(index)
	
	if (typeof(index) == "number") then
		
		return rawget(self.Values, index)
		
	else
		
		return rawget(TensorL, index)
		
	end
	
end

function TensorL:__newindex(index, value)
	
	rawset(self, index, value)
	
end

return TensorL
