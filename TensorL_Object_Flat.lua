local maximumTableLength = 2 ^ 26

local defaultMode = "Row"

local AqwamTensorLibrary = {}

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

local function getDimensionSizeArrayRecursive(tensor, targetDimensionSizeArray)

	if (type(tensor) ~= "table") then return end

	table.insert(targetDimensionSizeArray, #tensor)

	getDimensionSizeArrayRecursive(tensor[1], targetDimensionSizeArray)

end

local function getDimensionSizeArray(tensor)
	
	local dimensionSizeArray = {}

	getDimensionSizeArrayRecursive(tensor, dimensionSizeArray)

	return dimensionSizeArray
	
end

local function getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

	local totalSize = 1

	for _, value in ipairs(dimensionSizeArray) do totalSize = value * totalSize end

	return totalSize

end

local function convertTensorToData(tensor, dimensionSizeArray, numberOfDimensions, currentDimension, targetData, dataTableIndex, currentLinearIndex)

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do dataTableIndex, currentLinearIndex = convertTensorToData(tensor[i], dimensionSizeArray, numberOfDimensions, currentDimension + 1, targetData, dataTableIndex, currentLinearIndex) end

	else

		for _, value in ipairs(tensor) do 
			
			table.insert(targetData[dataTableIndex], value) 
			
			currentLinearIndex = currentLinearIndex + 1

			if ((currentLinearIndex % maximumTableLength) == 0) then dataTableIndex = dataTableIndex + 1 end
			
		end
		
	end
	
	return dataTableIndex, currentLinearIndex

end

local function setValueFromFunctionToData(functionToApply, dimensionSizeArray, numberOfDimensions, currentDimension, targetData, dataTableIndex, currentLinearIndex)

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do dataTableIndex, currentLinearIndex = setValueFromFunctionToData(functionToApply, dimensionSizeArray, numberOfDimensions, currentDimension + 1, targetData, dataTableIndex, currentLinearIndex) end

	else
		
		for i = 1, dimensionSizeArray[currentDimension], 1 do
			
			local value = functionToApply(currentLinearIndex)
			
			table.insert(targetData[dataTableIndex], value) 

			currentLinearIndex = currentLinearIndex + 1

			if ((currentLinearIndex % maximumTableLength) == 0) then dataTableIndex = dataTableIndex + 1 end
			
		end

	end

	return dataTableIndex, currentLinearIndex
	
end

local function generateEmptyDataFromDimensionSizeArray(dimensionSizeArray)
	
	local totalSize = getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

	local numberOfTablesForData = math.ceil(totalSize / maximumTableLength)
	
	local data = {}
	
	for i = 1, numberOfTablesForData, 1 do data[i] = {} end
	
	return data
	
end

function AqwamTensorLibrary.new(tensor, mode)

	local self = setmetatable({}, AqwamTensorLibrary)
	
	local dimensionSizeArray = getDimensionSizeArray(tensor)
	
	local data = generateEmptyDataFromDimensionSizeArray(dimensionSizeArray)

	convertTensorToData(tensor, dimensionSizeArray, #dimensionSizeArray, 1, data, 1, 1)
	
	self.data = data
	
	self.dimensionSizeArray = dimensionSizeArray
	
	self.mode = mode or defaultMode
	
	return self

end

function AqwamTensorLibrary.construct(data, dimensionSizeArray, mode)

	local self = setmetatable({}, AqwamTensorLibrary)

	self.data = data

	self.dimensionSizeArray = dimensionSizeArray
	
	self.mode = mode or defaultMode

	return self

end

function AqwamTensorLibrary.createTensor(dimensionSizeArray, initialValue, mode)

	initialValue = initialValue or 0

	local self = setmetatable({}, AqwamTensorLibrary)

	local data = generateEmptyDataFromDimensionSizeArray(dimensionSizeArray)
	
	setValueFromFunctionToData(function() return initialValue end, dimensionSizeArray, #dimensionSizeArray, 1, data, 1, 1)

	self.data = data

	self.dimensionSizeArray = dimensionSizeArray
	
	self.mode = mode or defaultMode

	return self

end

function AqwamTensorLibrary.createIdentityTensor(dimensionSizeArray, mode)

	local self = setmetatable({}, AqwamTensorLibrary)

	local data = generateEmptyDataFromDimensionSizeArray(dimensionSizeArray)
	
	local numberOfDimensions = #dimensionSizeArray
	
	local currentNumberOfOne = 1
	
	local linearIndex = 1
	
	local functionToApply = function(currentLinearIndex) -- Generalized row-major location calculation: (i−1)×(d2×d3×d4) + (i−1)×(d3×d4) + (i−1)×d4 + (i−1) + 1
		
		if (linearIndex ~= currentLinearIndex) then return 0 end
		
		currentNumberOfOne = currentNumberOfOne + 1
		
		local subtractedCurrentNumberOfOne = currentNumberOfOne - 1
		
		local multipliedDimensionSize = 1
		
		linearIndex = subtractedCurrentNumberOfOne
		
		for i = numberOfDimensions, 2, -1 do
			
			multipliedDimensionSize = multipliedDimensionSize * dimensionSizeArray[i]
			
			linearIndex = linearIndex + (multipliedDimensionSize * subtractedCurrentNumberOfOne)
			
		end
			
		linearIndex = linearIndex + 1 -- 1 is added due to the nature of Lua's 1-indexing.
		
		return 1
			
	end
	
	setValueFromFunctionToData(functionToApply, dimensionSizeArray, #dimensionSizeArray, 1, data, 1, 1)

	self.data = data

	self.dimensionSizeArray = dimensionSizeArray
	
	self.mode = mode or defaultMode

	return self

end

function AqwamTensorLibrary.createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation, mode)

	mean = mean or 0

	standardDeviation = standardDeviation or 1

	local self = setmetatable({}, AqwamTensorLibrary)
	
	local data = generateEmptyDataFromDimensionSizeArray(dimensionSizeArray)
	
	local functionToApply = function()
		
		local randomNumber1 = math.random()

		local randomNumber2 = math.random()

		local zScore = math.sqrt(-2 * math.log(randomNumber1)) * math.cos(2 * math.pi * randomNumber2) -- Box–Muller transform formula.

		return (zScore * standardDeviation) + mean		
		
	end

	setValueFromFunctionToData(functionToApply, dimensionSizeArray, #dimensionSizeArray, 1, data, 1, 1)

	self.data = data

	self.dimensionSizeArray = dimensionSizeArray
	
	self.mode = mode or defaultMode
	
	return self

end

function AqwamTensorLibrary.createRandomUniformTensor(dimensionSizeArray, minimumValue, maximumValue, mode)
	
	local self = setmetatable({}, AqwamTensorLibrary)

	local data = generateEmptyDataFromDimensionSizeArray(dimensionSizeArray)

	local functionToApply = function()
		
		if (minimumValue) and (maximumValue) then

			return math.random(minimumValue, maximumValue)

		elseif (minimumValue) and (not maximumValue) then

			return math.random(minimumValue)

		elseif (not minimumValue) and (not maximumValue) then

			return math.random()

		elseif (not minimumValue) and (maximumValue) then

			error("Invalid minimum value.")

		else

			error("An unknown error has occured when creating the random uniform tensor.")

		end
		
	end

	setValueFromFunctionToData(functionToApply, dimensionSizeArray, #dimensionSizeArray, 1, data, 1, 1)

	self.data = data

	self.dimensionSizeArray = dimensionSizeArray
	
	self.mode = mode or defaultMode

	return self
	
end

function AqwamTensorLibrary:getDimensionSizeArray()

	return deepCopyTable(self.dimensionSizeArray)

end

function AqwamTensorLibrary:getNumberOfDimensions()

	return #self.dimensionSizeArray

end

local function broadcast(tensor1, tensor2, deepCopyOriginalTensor)



end

function AqwamTensorLibrary:broadcast(tensor1, tensor2)

	local tensor1Value, tensor2Value = broadcast(tensor1.tensor, tensor2.tensor, true)

	return AqwamTensorLibrary.new(tensor1Value),  AqwamTensorLibrary.new(tensor2Value)

end

local function applyFunctionUsingOneTensor(functionToApply, tensor)

	local newData = {}

	for i = 1, #tensor, 1 do

		local data = tensor[i]

		local newSubData = {}

		for j, value in ipairs(data) do table.insert(newSubData, functionToApply(data[j])) end

		newData[i] = newSubData

	end

	return newData

end

local function applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2)

	local newData = {}
	
	for i = 1, #tensor1, 1 do
		
		local data1 = tensor1[i]
		
		local data2 = tensor2[i]

		local newSubData = {}

		for j, value in ipairs(data1) do table.insert(newSubData, functionToApply(data1[j], data2[j])) end

		newData[i] = newSubData
		
	end

	return newData

end

local function applyFunctionWhenTheFirstValueIsAScalar(functionToApply, scalar, tensor)

	local newData = {}

	for i = 1, #tensor, 1 do

		local data = tensor[i]

		local newSubData = {}

		for j, value in ipairs(data) do table.insert(newSubData, functionToApply(scalar, data[j])) end

		newData[i] = newSubData

	end

	return newData

end

local function applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, scalar)

	local newData = {}
	
	for i = 1, #tensor, 1 do

		local data = tensor[i]

		local newSubData = {}

		for j, value in ipairs(data) do table.insert(newSubData, functionToApply(data[j], scalar)) end

		newData[i] = newSubData

	end

	return newData

end

local function applyFunctionOnMultipleTensors(functionToApply, ...)

	local tensorArray = {...}

	local numberOfTensors = #tensorArray

	local tensor = tensorArray[1]

	if (numberOfTensors == 1) then 

		local dimensionSizeArray = getDimensionSizeArray(tensor)

		if (type(tensor) == "table") then

			return applyFunctionUsingOneTensor(functionToApply, tensor)

		else

			return functionToApply(tensor)

		end

	end

	for i = 2, numberOfTensors, 1 do

		local otherTensor = tensorArray[i]

		local isFirstValueATensor = (type(tensor) == "table")

		local isSecondValueATensor = (type(otherTensor) == "table")

		if (isFirstValueATensor) and (isSecondValueATensor) then

			--tensor, otherTensor = broadcast(tensor, otherTensor, false)

			tensor = applyFunctionUsingTwoTensors(functionToApply, tensor, otherTensor)

		elseif (not isFirstValueATensor) and (isSecondValueATensor) then

			tensor = applyFunctionWhenTheFirstValueIsAScalar(functionToApply, tensor, otherTensor)

		elseif (isFirstValueATensor) and (not isSecondValueATensor) then

			tensor = applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, otherTensor)

		else

			tensor = functionToApply(tensor, otherTensor)

		end

	end

	return tensor

end

function AqwamTensorLibrary:__add(other)

	local data, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a + b) end, self, other)

	return AqwamTensorLibrary.construct(data, dimensionSizeArray)

end

function AqwamTensorLibrary:add(...)

	local data, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a + b) end, self, ...)

	return AqwamTensorLibrary.construct(data, dimensionSizeArray)

end

function AqwamTensorLibrary:__sub(other)

	local data, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a - b) end, self, other)

	return AqwamTensorLibrary.construct(data, dimensionSizeArray)

end

function AqwamTensorLibrary:subtract(...)

	local data, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a - b) end, self, ...)

	return AqwamTensorLibrary.construct(data, dimensionSizeArray)

end

function AqwamTensorLibrary:__mul(other)

	local data, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a * b) end, self, other)

	return AqwamTensorLibrary.construct(data, dimensionSizeArray)

end

function AqwamTensorLibrary:multiply(...)

	local data, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a * b) end, self, ...)

	return AqwamTensorLibrary.construct(data, dimensionSizeArray)

end

function AqwamTensorLibrary:__div(other)

	local data, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a / b) end, self, other)

	return AqwamTensorLibrary.construct(data, dimensionSizeArray)

end

function AqwamTensorLibrary:divide(...)

	local data, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a / b) end, self, ...)

	return AqwamTensorLibrary.construct(data, dimensionSizeArray)

end

function AqwamTensorLibrary:__unm()

	local data, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a) return (-a) end, self)

	return AqwamTensorLibrary.construct(data, dimensionSizeArray)

end

function AqwamTensorLibrary:unaryMinus()

	local data, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a) return (-a) end, self)

	return AqwamTensorLibrary.construct(data, dimensionSizeArray)

end

function AqwamTensorLibrary:__index(index)

	if (typeof(index) == "number") then

		return rawget(self.data, index)

	else

		return rawget(AqwamTensorLibrary, index)

	end

end

function AqwamTensorLibrary:__newindex(index, value)

	rawset(self, index, value)

end

function AqwamTensorLibrary:__len()

	return #self.data

end

local function getLinearIndexForRowMajorStorage(dimensionIndexArray, dimensionSizeArray)
	
	local numberOfDimensions = #dimensionSizeArray
	
	if (#dimensionIndexArray ~= numberOfDimensions) then error("The number of dimensions does not match.") end

	for i, dimensionIndex in ipairs(dimensionIndexArray) do

		if (dimensionIndex <= 0) then error("The dimension index at dimension " .. i .. " must be greater than zero.") end

		if (dimensionIndex > dimensionSizeArray[i]) then error("The dimension index exceeds the dimension size at dimension " .. i .. ".") end

	end
	
	local linearIndex = 0

	local multipliedDimensionSize = 1

	for i = numberOfDimensions, 1, -1 do

		linearIndex = linearIndex + (multipliedDimensionSize * (dimensionIndexArray[i] - 1))

		multipliedDimensionSize = multipliedDimensionSize * dimensionSizeArray[i]

	end

	linearIndex = linearIndex + 1 -- 1 is added due to the nature of Lua's 1-indexing.
	
	return linearIndex
	
end

local function getLinearIndexForColumnMajorStorage(dimensionIndexArray, dimensionSizeArray)

	local numberOfDimensions = #dimensionSizeArray

	if (#dimensionIndexArray ~= numberOfDimensions) then error("The number of dimensions does not match.") end

	for i, dimensionIndex in ipairs(dimensionIndexArray) do

		if (dimensionIndex <= 0) then error("The dimension index at dimension " .. i .. " must be greater than zero.") end

		if (dimensionIndex > dimensionSizeArray[i]) then error("The dimension index exceeds the dimension size at dimension " .. i .. ".") end

	end

	local linearIndex = 0

	local multipliedDimensionSize = 1

	for i = 1, numberOfDimensions, 1 do
		
		linearIndex = linearIndex + (multipliedDimensionSize * (dimensionIndexArray[i] - 1))
		
		multipliedDimensionSize = multipliedDimensionSize * dimensionSizeArray[i]
		
	end

	linearIndex = linearIndex + 1 -- 1 is added due to the nature of Lua's 1-indexing.

	return linearIndex

end

local getLinearIndexFunctionList = {
	
	["Row"] = getLinearIndexForRowMajorStorage,
	
	["Column"] = getLinearIndexForColumnMajorStorage
	
}

local function getDataIndex(linearIndex)
	
	local dataTableIndex = math.ceil(linearIndex / maximumTableLength)

	local dataIndex = linearIndex % maximumTableLength
	
	return dataTableIndex, dataIndex
	
end

function AqwamTensorLibrary:setValue(value, dimensionIndexArray)
	
	local linearIndex = getLinearIndexFunctionList[self.mode](dimensionIndexArray, self.dimensionSizeArray)
	
	local dataTableIndex, dataIndex = getDataIndex(linearIndex)
	
	self.data[dataTableIndex][dataIndex] = value
	
end

function AqwamTensorLibrary:getValue(dimensionIndexArray)

	local linearIndex = getLinearIndexFunctionList[self.mode](dimensionIndexArray, self.dimensionSizeArray)

	local dataTableIndex, dataIndex = getDataIndex(linearIndex)
	
	return self.data[dataTableIndex][dataIndex]

end

function AqwamTensorLibrary:transpose(dimensionArray)
	
	if (#dimensionArray ~= 2) then error("Dimension array must contain 2 dimensions.") end

	local dimension1 = dimensionArray[1]

	local dimension2 = dimensionArray[2]

	local numberOfDimensions = #self.dimensionSizeArray

	if (dimension1 <= 0) then error("The first dimension must be greater than zero.") end

	if (dimension2 <= 0) then error("The second dimension must be greater than zero.") end

	if (dimension1 > numberOfDimensions) then error("The first dimension exceeds the tensor's number of dimensions") end

	if (dimension2 > numberOfDimensions) then error("The second dimension exceeds the tensor's number of dimensions") end

	if (dimension1 == dimension2) then error("The first dimension is equal to the second dimension.") end
	
end

function AqwamTensorLibrary:destroy()

	self.data = nil
	
	self.dimensionSizeArray = nil

	setmetatable(self, nil)

end

return AqwamTensorLibrary
