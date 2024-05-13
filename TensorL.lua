local AqwamTensorLibrary = {}

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
	
	for i = 1, #tensor1 do  
		
		if (#dimensionArray1 > 1) then

			result[i] = applyOperation(operation, tensor1[i], tensor2[i])

		else

			result[i] = operation(tensor1[i], tensor2[i])

		end
		
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

local function fullSum(tensor)
	
	local dimensionArray = getDimensionArray(tensor)

	local numberOfValues = dimensionArray[1]

	local result = 0
	
	for i = 1, numberOfValues, 1 do 
		
		if (#dimensionArray > 1) then

			result += fullSum(tensor[i]) 

		else

			result += tensor[i]
			
		end
		
	end
	
	return result
	
end

local function dimensionSumRecursive(result, tensor, dimension)
	
	local dimensionArray = getDimensionArray(tensor)
	
	local numberOfDimensions = #dimensionArray
	
	local numberOfValues = dimensionArray[1]
	
	for i = 1, numberOfValues, 1 do 	
		
		if (numberOfDimensions == dimension) then

			dimensionSumRecursive(result[i], tensor[i], dimension)

		else

			result[i] += tensor[i]

		end
		
	end

end

local function accumulateDimension(result, tensor, dimension, dimensionIndices, dimensionArray)
	if dimension == #dimensionIndices then
		for i = 1, dimensionArray[dimensionIndices[dimension]], 1 do
			local indices = {}
			for j = 1, #dimensionIndices do
				indices[j] = dimensionIndices[j]
			end
			indices[dimension] = i
			result[table.unpack(indices)] = (result[table.unpack(indices)] or 0) + tensor[table.unpack(indices)]
		end
	else
		for i = 1, dimensionArray[dimensionIndices[dimension]], 1 do
			dimensionIndices[dimension] = i
			accumulateDimension(result, tensor, dimension + 1, dimensionIndices, dimensionArray)
		end
	end
end

local function dimSumRecursive(result, tensor, targetDimension)
	
	local dimensionArray = getDimensionArray(tensor)
	
	local currentDimension = #dimensionArray
	
	local numberOfValues = dimensionArray[1]
	
	for i = 1, numberOfValues, 1 do
		
		if (currentDimension == targetDimension) then
			
			print(getDimensionArray(result))
			
			print(getDimensionArray(tensor))
			
			result[i] += tensor[i]

		else

			dimensionSumRecursive(result[i], tensor[i], targetDimension)

		end
		
	end
	
end

local function dimensionSum(tensor, targetDimension)
	
	local dimensionArray = getDimensionArray(tensor)
	
	local newDimensionArray = deepCopyTable(dimensionArray)
	
	dimensionArray[targetDimension] = 1
	
	local result = createTensor(dimensionArray, 0)

	dimSumRecursive(result, tensor, targetDimension)
	
	--[[

	for dimension1 = 1, dimensionArray[1], 1 do

		for dimension2 = 1, dimensionArray[2], 1 do

			for dimension3 = 1, dimensionArray[3], 1 do

				if (dimension == 1) then

					result[1][dimension2][dimension3] += tensor[dimension1][dimension2][dimension3]	

				elseif (dimension == 2) then

					result[dimension1][1][dimension3] += tensor[dimension1][dimension2][dimension3]

				elseif (dimension == 3) then

					result[dimension1][dimension2][1] += tensor[dimension1][dimension2][dimension3]

				else

					error("Invalid dimension.")

				end 

			end

		end	

	end
	
	--]]
	
	return result
	
end

local function sum(tensor, dimension)
	
	if not dimension then return fullSum(tensor) end
	
	local numberOfDimension = getNumberOfDimensions(tensor)
	
	if (dimension > numberOfDimension) or (dimension < 1) then error("Invalid dimensions.") end
	
	local reversedSequence = {}
	
	for i = numberOfDimension, 1, -1 do table.insert(reversedSequence, i) end
	
	local selectedDimension = reversedSequence[dimension]
	
	return dimensionSum(tensor, selectedDimension)
	
end

local function tensorProduct(tensor1, tensor2)
	
	local dimensionArray1 = getDimensionArray(tensor1)
	
	local dimensionArray2 = getDimensionArray(tensor2)

	for i, _ in ipairs(dimensionArray1) do if (dimensionArray1[i] ~= dimensionArray2[i]) then error("Invalid dimensions.") end end

	local numberOfValues = dimensionArray1[1]
	
	local result = {}
	
	for i = 1, numberOfValues, 1 do

		if (#dimensionArray1 > 1) then

			local subproduct = tensorProduct(tensor1[i], tensor2[i])

			table.insert(result, subproduct)

		else

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
	
	for i = 1, numberOfValues, 1 do  
		
		if (#dimensionArray1 > 1) then

			result += innerProduct(tensor1[i], tensor2[i])

		else

			result += (tensor1[i] * tensor2[i])

		end
		
	end

	return result
	
end

local function outerProduct(tensor1, tensor2)
	
	local dimensionArray1 = getDimensionArray(tensor1)
	
	local dimensionArray2 = getDimensionArray(tensor2)

	for i, _ in ipairs(dimensionArray1) do if dimensionArray1[i] ~= dimensionArray2[i] then error("Invalid dimensions.") end end

	local numberOfValues = dimensionArray1[1]
	
	local result = {}
	
	for i = 1, numberOfValues do

		if (#dimensionArray1 > 1) then

			result[i] = outerProduct(tensor1[i], tensor2[i])

		else

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

local function transpose(tensor, dimension1, dimension2)
	
	local dimensionArray = getDimensionArray(tensor)
	
	local numberOfDimensions = #dimensionArray
	
end

function AqwamTensorLibrary.new(...)
	
	local self = setmetatable({}, AqwamTensorLibrary)

	self.Values = ...

	return self
	
end

function AqwamTensorLibrary.create(dimensionArray, initialValue)
	
	initialValue = initialValue or 0
	
	local self = setmetatable({}, AqwamTensorLibrary)
	
	self.Values = createTensor(dimensionArray, initialValue)
	
	return self
	
end

function AqwamTensorLibrary:broadcast(dimensionsArray, values)

	local isNumber = typeof(values) == "number"

	if isNumber then return self.create(dimensionsArray, values) end
	
	return values

end

function AqwamTensorLibrary:getNumberOfDimensions()

	return getNumberOfDimensions(self)

end

function AqwamTensorLibrary:getDimensionArray()
	
	return getDimensionArray(self)
	
end

function AqwamTensorLibrary:print()

	print(self)
	
end

function AqwamTensorLibrary:transpose(dimension1, dimension2)
	
	if (typeof(dimension1) ~= "number") or (typeof(dimension2) ~= "number") then error("Dimensions are not numbers.") end
	
	local numberOfDimension = getNumberOfDimensions(self)

	if (dimension1 < 1) or (dimension1 > numberOfDimension) or (dimension2 < 1) or (dimension2 > numberOfDimension) or (dimension1 == dimension2) then error("Invalid dimensions.") end
	
	local result = transpose(self, dimension1, dimension2)

	return result
	
end

function AqwamTensorLibrary:__eq(other)
	
	local numberOfDimensions1 = getNumberOfDimensions(self)

	local numberOfDimensions2 = getNumberOfDimensions(other)

	if (numberOfDimensions1 ~= numberOfDimensions2) then return false end
	
	local operation = function(a, b) return (a == b) end

	local result = applyOperation(operation, self, other)
	
	local isEqual = eq(result)

	return isEqual
	
end

function AqwamTensorLibrary:isEqualTo(other)

	local numberOfDimensions1 = getNumberOfDimensions(self)

	local numberOfDimensions2 = getNumberOfDimensions(other)
	
	if (numberOfDimensions1 ~= numberOfDimensions2) then error("Invalid dimensions.") end

	local operation = function(a, b) return (a == b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)

end

function AqwamTensorLibrary:isGreaterThan(other)

	local numberOfDimensions1 = getNumberOfDimensions(self)

	local numberOfDimensions2 = getNumberOfDimensions(other)

	if (numberOfDimensions1 ~= numberOfDimensions2) then error("Invalid dimensions.") end

	local operation = function(a, b) return (a > b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)

end

function AqwamTensorLibrary:isGreaterOrEqualTo(other)

	local numberOfDimensions1 = getNumberOfDimensions(self)

	local numberOfDimensions2 = getNumberOfDimensions(other)

	if (numberOfDimensions1 ~= numberOfDimensions2) then error("Invalid dimensions.") end

	local operation = function(a, b) return (a >= b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)

end

function AqwamTensorLibrary:isLessThan(other)

	local numberOfDimensions1 = getNumberOfDimensions(self)

	local numberOfDimensions2 = getNumberOfDimensions(other)

	if (numberOfDimensions1 ~= numberOfDimensions2) then error("Invalid dimensions.") end
	
	local operation = function(a, b) return (a < b) end

	local result = applyOperation(operation, self, other)
	
	return self.new(result)

end

function AqwamTensorLibrary:isLessOrEqualTo(other)

	local numberOfDimensions1 = getNumberOfDimensions(self)

	local numberOfDimensions2 = getNumberOfDimensions(other)

	if (numberOfDimensions1 ~= numberOfDimensions2) then error("Invalid dimensions.") end

	local operation = function(a, b) return (a <= b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)

end

function AqwamTensorLibrary:sum(dimension)
	
	local result = sum(self, dimension)
	
	if not dimension then return result end
	
	return result
	
end

function AqwamTensorLibrary:tensorProduct(other)
	
	local result = tensorProduct(self, other)
	
	return self.new(result)
	
end

function AqwamTensorLibrary:innerProduct(other)

	return innerProduct(self, other)

end

function AqwamTensorLibrary:outerProduct(other)
	
	local result = outerProduct(self, other)

	return self.create(result)

end

function AqwamTensorLibrary:copy()
	
	return deepCopyTable(self)
	
end

function AqwamTensorLibrary:rawCopy()
	
	return deepCopyTable(self.Values)
	
end

function AqwamTensorLibrary:__add(other)
	
	local numberOfDimensions = self:getNumberOfDimensions()
	
	local other = self:broadcast(numberOfDimensions, other)
	
	local operation = function(a, b) return (a + b) end
	
	local result = applyOperation(operation, self, other)

	return self.new(result)
	
end

function AqwamTensorLibrary:__sub(other)
	
	local other = self:broadcast(other, table.unpack(self:getNumberOfDimensions()))

	local operation = function(a, b) return (a - b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)
	
end

function AqwamTensorLibrary:__mul(other)
	
	local other = self:broadcast(other, table.unpack(self:getNumberOfDimensions()))

	local operation = function(a, b) return (a * b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)
	
end

function AqwamTensorLibrary:__div(other)
	
	local other = self:broadcast(other, table.unpack(self:getNumberOfDimensions()))

	local operation = function(a, b) return (a / b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)
	
end

function AqwamTensorLibrary:__unm(other)
	
	local other = self:broadcast(-1, table.unpack(self:getNumberOfDimensions()))

	local operation = function(a, b) return (a * b) end

	local result = applyOperation(operation, self, other)

	return self.new(result)
	
end

function AqwamTensorLibrary:__tostring()
	
	local text = "\n\n" .. createString(self) .. "\n\n"

	return text
	
end

function AqwamTensorLibrary:__len()
	
	return #self.Values
	
end

function AqwamTensorLibrary:__index(index)
	
	if (typeof(index) == "number") then
		
		return rawget(self.Values, index)
		
	else
		
		return rawget(AqwamTensorLibrary, index)
		
	end
	
end

function AqwamTensorLibrary:__newindex(index, value)
	
	rawset(self, index, value)
	
end

return AqwamTensorLibrary
