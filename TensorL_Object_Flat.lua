local maximumTableLength = 2 ^ 26

local defaultMode = "Row"

local AqwamTensorLibrary = {}

local function checkIfDimensionIndexArrayAreEqual(dimensionSizeArray1, dimensionSizeArray2)

	if (#dimensionSizeArray1 ~= #dimensionSizeArray2) then return false end

	for i, index in ipairs(dimensionSizeArray1) do

		if (index ~= dimensionSizeArray2[i]) then return false end

	end

	return true

end

local function checkIfValueIsOutOfBounds(value, minimumValue, maximumValue)

	return (value < minimumValue) or (value > maximumValue)

end

local function throwErrorIfDimensionSizeIndexIsOutOfBounds(dimensionSizeIndex, minimumDimensionSizeIndex, maximumDimensionSizeIndex)

	if checkIfValueIsOutOfBounds(dimensionSizeIndex, minimumDimensionSizeIndex, maximumDimensionSizeIndex) then error("The dimension size index is out of bounds.") end

end

local function throwErrorIfDimensionIsOutOfBounds(dimension, minimumNumberOfDimensions, maximumNumberOfDimensions)

	if checkIfValueIsOutOfBounds(dimension, minimumNumberOfDimensions, maximumNumberOfDimensions) then error("The dimension is out of bounds.") end

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

local function convertTensorToData(tensor, dimensionSizeArray, numberOfDimensions, currentDimension, targetData, dataTableTableIndex, dataTableIndex, currentLinearIndex)

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do dataTableTableIndex, dataTableIndex, currentLinearIndex = convertTensorToData(tensor[i], dimensionSizeArray, numberOfDimensions, currentDimension + 1, targetData, dataTableTableIndex, dataTableIndex, currentLinearIndex) end

	else

		for _, value in ipairs(tensor) do 
			
			table.insert(targetData[dataTableTableIndex][dataTableIndex], value) 
			
			currentLinearIndex = currentLinearIndex + 1

			if ((currentLinearIndex % maximumTableLength) == 0) then dataTableIndex = dataTableIndex + 1 end
			
			if ((dataTableIndex % maximumTableLength) == 0) then dataTableTableIndex = dataTableTableIndex + 1 end
			
		end
		
	end
	
	return dataTableTableIndex, dataTableIndex, currentLinearIndex

end

local function setValueFromFunctionToData(functionToApply, dimensionSizeArray, numberOfDimensions, currentDimension, targetData, dataTableTableIndex, dataTableIndex, currentLinearIndex)

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do dataTableTableIndex, dataTableIndex, currentLinearIndex = setValueFromFunctionToData(functionToApply, dimensionSizeArray, numberOfDimensions, currentDimension + 1, targetData, dataTableTableIndex, dataTableIndex, currentLinearIndex) end

	else
		
		for i = 1, dimensionSizeArray[currentDimension], 1 do
			
			local value = functionToApply(currentLinearIndex)
			
			table.insert(targetData[dataTableTableIndex][dataTableIndex], value) 

			currentLinearIndex = currentLinearIndex + 1

			if ((currentLinearIndex % maximumTableLength) == 0) then dataTableIndex = dataTableIndex + 1 end
			
			if ((dataTableIndex % maximumTableLength) == 0) then dataTableTableIndex = dataTableTableIndex + 1 end
			
		end

	end

	return dataTableTableIndex, dataTableIndex, currentLinearIndex
	
end

local function createEmptyDataFromDimensionSizeArray(dimensionSizeArray)
	
	local totalSize = getTotalSizeFromDimensionSizeArray(dimensionSizeArray)
	
	local squaredMaximumTableLength = math.pow(maximumTableLength, 2)
	
	local subDataIndex = 1
	
	local subDataDataIndex = 1
	
	local data = {}

	data[subDataIndex] = {}

	data[subDataIndex][subDataDataIndex] = {}
	
	for i = totalSize, 1, 1 do
		
		if ((totalSize % math.pow(squaredMaximumTableLength)) == 0) then
			
			subDataIndex = subDataIndex + 1
			
			subDataDataIndex = 1
			
			data[subDataIndex] = {}
			
		end
		
		if ((totalSize % maximumTableLength) == 0) then
			
			subDataDataIndex = subDataDataIndex + 1

			data[subDataIndex][subDataDataIndex] = {}

		end
		
	end
	
	return data
	
end

function AqwamTensorLibrary.new(tensor, mode)

	local self = setmetatable({}, AqwamTensorLibrary)
	
	local dimensionSizeArray = getDimensionSizeArray(tensor)
	
	local data = createEmptyDataFromDimensionSizeArray(dimensionSizeArray)

	convertTensorToData(tensor, dimensionSizeArray, #dimensionSizeArray, 1, data, 1, 1, 1)
	
	self.data = data
	
	self.dimensionSizeArray = dimensionSizeArray
	
	self.mode = mode or defaultMode
	
	return self

end

function AqwamTensorLibrary.construct(data, dimensionSizeArray, mode)

	local self = setmetatable({}, AqwamTensorLibrary)

	self.data = data or {{{}}}

	self.dimensionSizeArray = dimensionSizeArray or {}
	
	self.mode = mode or defaultMode

	return self

end

function AqwamTensorLibrary.convertToObject(convertedTable)
	
	local self = setmetatable({}, AqwamTensorLibrary)
	
	self.data = convertedTable.data or {{{}}}

	self.dimensionSizeArray = convertedTable.dimensionSizeArray or {}

	self.mode = convertedTable.mode or defaultMode
	
end

function AqwamTensorLibrary:convertToTable()
	
	local convertedTable = {
		
		data = deepCopyTable(self.data),
		
		dimensionSizeArray = deepCopyTable(self.dimensionSizeArray),
		
		mode = deepCopyTable(self.mode)
		
	}
	
	return convertedTable
	
end

function AqwamTensorLibrary.createTensor(dimensionSizeArray, initialValue, mode)

	initialValue = initialValue or 0

	local self = setmetatable({}, AqwamTensorLibrary)

	local data = createEmptyDataFromDimensionSizeArray(dimensionSizeArray)
	
	setValueFromFunctionToData(function() return initialValue end, dimensionSizeArray, #dimensionSizeArray, 1, data, 1, 1, 1)

	self.data = data

	self.dimensionSizeArray = dimensionSizeArray
	
	self.mode = mode or defaultMode

	return self

end

function AqwamTensorLibrary.createIdentityTensor(dimensionSizeArray, mode)

	local self = setmetatable({}, AqwamTensorLibrary)

	local data = createEmptyDataFromDimensionSizeArray(dimensionSizeArray)
	
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
	
	setValueFromFunctionToData(functionToApply, dimensionSizeArray, #dimensionSizeArray, 1, data, 1, 1, 1)

	self.data = data

	self.dimensionSizeArray = dimensionSizeArray
	
	self.mode = mode or defaultMode

	return self

end

function AqwamTensorLibrary.createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation, mode)

	mean = mean or 0

	standardDeviation = standardDeviation or 1

	local self = setmetatable({}, AqwamTensorLibrary)
	
	local data = createEmptyDataFromDimensionSizeArray(dimensionSizeArray)
	
	local functionToApply = function()
		
		local randomNumber1 = math.random()

		local randomNumber2 = math.random()

		local zScore = math.sqrt(-2 * math.log(randomNumber1)) * math.cos(2 * math.pi * randomNumber2) -- Box–Muller transform formula.

		return (zScore * standardDeviation) + mean		
		
	end

	setValueFromFunctionToData(functionToApply, dimensionSizeArray, #dimensionSizeArray, 1, data, 1, 1, 1)

	self.data = data

	self.dimensionSizeArray = dimensionSizeArray
	
	self.mode = mode or defaultMode
	
	return self

end

function AqwamTensorLibrary.createRandomUniformTensor(dimensionSizeArray, minimumValue, maximumValue, mode)
	
	local self = setmetatable({}, AqwamTensorLibrary)

	local data = createEmptyDataFromDimensionSizeArray(dimensionSizeArray)

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

local function throwErrorWhenDimensionIndexArrayIsOutOfBounds(dimensionIndexArray, dimensionSizeArray)

	if (#dimensionIndexArray ~= #dimensionSizeArray) then error("The number of dimensions does not match.") end

	for i, dimensionIndex in ipairs(dimensionIndexArray) do

		if (dimensionIndex <= 0) then error("The dimension index at dimension " .. i .. " must be greater than zero.") end

		if (dimensionIndex > dimensionSizeArray[i]) then error("The dimension index exceeds the dimension size at dimension " .. i .. ".") end

	end

end


local function getLinearIndexForRowMajorStorage(dimensionIndexArray, dimensionSizeArray)

	throwErrorWhenDimensionIndexArrayIsOutOfBounds(dimensionIndexArray, dimensionSizeArray)

	local linearIndex = 0

	local multipliedDimensionSize = 1

	for i = #dimensionSizeArray, 1, -1 do

		linearIndex = linearIndex + (multipliedDimensionSize * (dimensionIndexArray[i] - 1))

		multipliedDimensionSize = multipliedDimensionSize * dimensionSizeArray[i]

	end

	linearIndex = linearIndex + 1 -- 1 is added due to the nature of Lua's 1-indexing.

	return linearIndex

end

local function getLinearIndexForColumnMajorStorage(dimensionIndexArray, dimensionSizeArray)

	throwErrorWhenDimensionIndexArrayIsOutOfBounds(dimensionIndexArray, dimensionSizeArray)

	local linearIndex = 0

	local multipliedDimensionSize = 1

	for i = 1, #dimensionSizeArray, 1 do

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

local function incrementDimensionIndexArray(dimensionSizeArray, dimensionIndexArray)

	for i = #dimensionIndexArray, 1, -1 do

		dimensionIndexArray[i] = dimensionIndexArray[i] + 1

		if (dimensionIndexArray[i] <= dimensionSizeArray[i]) then break end

		dimensionIndexArray[i] = 1

	end

	return dimensionIndexArray

end

local function getDataIndex(linearIndex)

	local subSubDataIndex = (linearIndex - 1) % maximumTableLength + 1

	local subDataIndex = math.floor((linearIndex - 1) / maximumTableLength) % maximumTableLength + 1

	local dataIndex = math.floor((linearIndex - 1) / (maximumTableLength * maximumTableLength)) + 1

	return dataIndex, subDataIndex, subSubDataIndex

end

local function applyFunctionUsingOneTensor(functionToApply, tensor)

	local newData = {}
	
	for _, subData in ipairs(tensor.data) do 

		local newSubData = {}

		for _, subSubData in ipairs(subData) do 

			local newSubSubData = {}

			for _, value in ipairs(subSubData) do table.insert(newSubSubData, functionToApply(value)) end

			table.insert(newSubData, newSubSubData)

		end

		table.insert(newData, newSubData)

	end

	return newData

end

local function applyFunctionUsingTwoTensorsOfSameModes(functionToApply, tensor1, tensor2)
	
	local newData = {}
	
	for i, subData1 in ipairs(tensor1.data) do 
		
		local subData2 = tensor2[i]

		local newSubData = {}

		for j, subSubData1 in ipairs(subData1) do 
			
			local subSubData2 = subData2[j]

			local newSubSubData = {}

			for k, value in ipairs(subSubData1) do table.insert(newSubSubData, functionToApply(value, subSubData2[k])) end

			table.insert(newSubData, newSubSubData)

		end

		table.insert(newData, newSubData)

	end
	
	return AqwamTensorLibrary.construct(newData, deepCopyTable(tensor1.dimensionSizeArray), deepCopyTable(tensor1.mode))
	
end

local function applyFunctionUsingTwoTensorsOfDifferentModes(functionToApply, tensor1, tensor2, dimensionSizeArray)
	
	local currentDimensionIndexArray = table.create(#dimensionSizeArray, 1)
	
	local getLinearIndex1 = getLinearIndexFunctionList[tensor1.mode]
	
	local getLinearIndex2 = getLinearIndexFunctionList[tensor2.mode] 
	
	local newData = createEmptyDataFromDimensionSizeArray(dimensionSizeArray)
	
	repeat
		
		local linearIndex1 = getLinearIndex1(currentDimensionIndexArray)

		local linearIndex2 = getLinearIndex2(currentDimensionIndexArray)

		local dataIndex1, subDataIndex1, subSubDataIndex1 = getDataIndex(linearIndex1)

		local dataIndex2, subDataIndex2, subSubDataIndex2 = getDataIndex(linearIndex2)

		newData[dataIndex1][subDataIndex1][subSubDataIndex1] = functionToApply(tensor1[dataIndex1][subDataIndex1][subSubDataIndex1], tensor1[dataIndex2][dataIndex2][dataIndex2])

		currentDimensionIndexArray = incrementDimensionIndexArray(dimensionSizeArray, currentDimensionIndexArray)
		
	until checkIfDimensionIndexArrayAreEqual(currentDimensionIndexArray, dimensionSizeArray)

	return AqwamTensorLibrary.construct(newData, deepCopyTable(tensor1.dimensionSizeArray), deepCopyTable(tensor1.mode))
	
end

local function applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2, dimensionSizeArray, index)
	
	local tensor1Mode = tensor1.mode
	
	local tensor2Mode = tensor2.mode
	
	if (tensor1Mode ~= "Row") and (tensor1Mode ~= "Column") then error("Tensor " .. (index - 1) .. " contains an invalid mode.") end -- Index is subtracted by one because it starts at 2 instead of 1.
	
	if (tensor2Mode ~= "Row") and (tensor2Mode ~= "Column") then error("Tensor " .. index .. " contains an invalid mode.") end
	
	if (tensor1Mode == tensor2Mode) then return applyFunctionUsingTwoTensorsOfSameModes(functionToApply, tensor1, tensor2) end

	return applyFunctionUsingTwoTensorsOfDifferentModes(functionToApply, tensor1, tensor2, dimensionSizeArray)

end

local function applyFunctionWhenTheFirstValueIsAScalar(functionToApply, scalar, tensor)

	local newData = {}

	for _, subData in ipairs(tensor.data) do 

		local newSubData = {}

		for _, subSubData in ipairs(subData) do 

			local newSubSubData = {}

			for _, value in ipairs(subSubData) do table.insert(newSubSubData, functionToApply(scalar, value)) end

			table.insert(newSubData, newSubSubData)

		end

		table.insert(newData, newSubData)

	end
	
	return AqwamTensorLibrary.construct(newData, deepCopyTable(tensor.dimensionSizeArray), deepCopyTable(tensor.mode))

end

local function applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, scalar)

	local newData = {}
	
	for _, subData in ipairs(tensor.data) do 
		
		local newSubData = {}

		for _, subSubData in ipairs(subData) do 
			
			local newSubSubData = {}

			for _, value in ipairs(subSubData) do table.insert(newSubSubData, functionToApply(value, scalar)) end
			
			table.insert(newSubData, newSubSubData)

		end
		
		table.insert(newData, newSubData)

	end
	
	return AqwamTensorLibrary.construct(newData, deepCopyTable(tensor.dimensionSizeArray), deepCopyTable(tensor.mode))

end

local function applyFunctionOnMultipleTensors(functionToApply, ...)

	local tensorArray = {...}

	local numberOfTensors = #tensorArray

	local tensor = tensorArray[1]

	if (numberOfTensors == 1) then 

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

			tensor = applyFunctionUsingTwoTensors(functionToApply, tensor, otherTensor, dimensionSizeArray, i)

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

function AqwamTensorLibrary:applyFunction(functionToApply)

	return applyFunctionOnMultipleTensors(functionToApply, self)

end

function AqwamTensorLibrary:__add(other)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a + b) end, self, other)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:add(...)

	local functionToApply = function(a, b) return (a + b) end

	if (self.dimensionSizeArray) then

		return applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		return applyFunctionOnMultipleTensors(functionToApply, ...)

	end

end

function AqwamTensorLibrary:__sub(other)

	return applyFunctionOnMultipleTensors(function(a, b) return (a - b) end, self, other)

end

function AqwamTensorLibrary:subtract(...)

	local functionToApply = function(a, b) return (a - b) end

	if (self.dimensionSizeArray) then

		return applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		return applyFunctionOnMultipleTensors(functionToApply, ...)

	end

end

function AqwamTensorLibrary:__mul(other)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a * b) end, self, other)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:multiply(...)

	local functionToApply = function(a, b) return (a * b) end

	local resultTensor

	if (self.dimensionSizeArray) then

		return applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		return applyFunctionOnMultipleTensors(functionToApply, ...)

	end

end

function AqwamTensorLibrary:__div(other)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a / b) end, self, other)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:divide(...)

	local functionToApply = function(a, b) return (a / b) end

	if (self.dimensionSizeArray) then

		return applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		return applyFunctionOnMultipleTensors(functionToApply, ...)

	end

end

function AqwamTensorLibrary:__unm()

	return applyFunctionOnMultipleTensors(function(a) return (-a) end, self)

end

function AqwamTensorLibrary:unaryMinus(...)

	local functionToApply = function(a) return (-a) end

	if (self.dimensionSizeArray) then

		return applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		return applyFunctionOnMultipleTensors(functionToApply, ...)

	end

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

function AqwamTensorLibrary:setValue(value, dimensionIndexArray)
	
	local linearIndex = getLinearIndexFunctionList[self.mode](dimensionIndexArray, self.dimensionSizeArray)
	
	local dataIndex, subDataIndex, subSubDataIndex = getDataIndex(linearIndex)
	
	self.data[dataIndex][subDataIndex][subSubDataIndex] = value
	
end

function AqwamTensorLibrary:getValue(dimensionIndexArray)

	local linearIndex = getLinearIndexFunctionList[self.mode](dimensionIndexArray, self.dimensionSizeArray)

	local dataIndex, subDataIndex, subSubDataIndex = getDataIndex(linearIndex)
	
	return self.data[dataIndex][subDataIndex][subSubDataIndex]

end

function AqwamTensorLibrary:transpose(dimensionArray)
	
	local dimensionSizeArray = self.dimensionSizeArray
	
	if (#dimensionArray ~= 2) then error("Dimension array must contain 2 dimensions.") end

	local dimension1 = dimensionArray[1]

	local dimension2 = dimensionArray[2]

	local numberOfDimensions = #dimensionSizeArray

	if (dimension1 <= 0) then error("The first dimension must be greater than zero.") end

	if (dimension2 <= 0) then error("The second dimension must be greater than zero.") end

	if (dimension1 > numberOfDimensions) then error("The first dimension exceeds the tensor's number of dimensions") end

	if (dimension2 > numberOfDimensions) then error("The second dimension exceeds the tensor's number of dimensions") end

	if (dimension1 == dimension2) then error("The first dimension is equal to the second dimension.") end

	local newDimensionSizeArray = table.clone(dimensionSizeArray)

	newDimensionSizeArray[dimension1] = dimensionSizeArray[dimension2]

	newDimensionSizeArray[dimension2] = dimensionSizeArray[dimension1]
	
	local getLinearIndex = getLinearIndexFunctionList[self.mode]
	
	local currentDimensionIndexArray = table.create(numberOfDimensions, 1)
	
	local data = self.data
	
	local newData = createEmptyDataFromDimensionSizeArray(newDimensionSizeArray)
	
	repeat
		
		local targetDimensionIndexArray = table.clone(currentDimensionIndexArray)

		targetDimensionIndexArray[dimension1] = currentDimensionIndexArray[dimension2]

		targetDimensionIndexArray[dimension2] = currentDimensionIndexArray[dimension1]
		
		local currentLinearIndex = getLinearIndex(currentDimensionIndexArray, dimensionSizeArray)

		local targetLinearIndex = getLinearIndex(targetDimensionIndexArray, newDimensionSizeArray)
		
		local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

		local targetDataIndex, targetSubDataIndex, targetSubSubDataIndex = getDataIndex(targetLinearIndex)

		newData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] = data[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex]

		currentDimensionIndexArray = incrementDimensionIndexArray(dimensionSizeArray, currentDimensionIndexArray)
		
	until checkIfDimensionIndexArrayAreEqual(currentDimensionIndexArray, dimensionSizeArray)
	
	return AqwamTensorLibrary.construct(newData, newDimensionSizeArray, self.mode)
	
end

local function sumFromAllDimensionsFromData(data)

	local totalValue = 0
	
	for _, subData in ipairs(data) do 

		for _, subSubData in ipairs(subData) do 
			
			for _, value in ipairs(subSubData) do totalValue = totalValue + value end
			
		end

	end

	return totalValue

end

function AqwamTensorLibrary:sum(dimension)
	
	local data = self.data

	if (not dimension) then return sumFromAllDimensionsFromData(data) end

	if (type(dimension) ~= "number") then error("The dimension must be a number.") end
	
	local dimensionSizeArray = self.dimensionSizeArray

	local numberOfDimensions = #dimensionSizeArray

	throwErrorIfDimensionIsOutOfBounds(dimension, 1, numberOfDimensions)
	
	local newDimensionSizeArray = table.clone(dimensionSizeArray)

	newDimensionSizeArray[dimension] = 1
	
	local mode = self.mode

	local getLinearIndex = getLinearIndexFunctionList[mode]

	local currentDimensionIndexArray = table.create(numberOfDimensions, 1)

	local newData = createEmptyDataFromDimensionSizeArray(newDimensionSizeArray)
	
	repeat

		local targetDimensionIndexArray = table.clone(currentDimensionIndexArray)

		targetDimensionIndexArray[dimension] = 1

		local currentLinearIndex = getLinearIndex(currentDimensionIndexArray, dimensionSizeArray)

		local targetLinearIndex = getLinearIndex(targetDimensionIndexArray, newDimensionSizeArray)

		local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

		local targetDataIndex, targetSubDataIndex, targetSubSubDataIndex = getDataIndex(targetLinearIndex)
		
		local newDataValue = newData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] or 0

		newData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] = newDataValue + data[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex]

		currentDimensionIndexArray = incrementDimensionIndexArray(dimensionSizeArray, currentDimensionIndexArray)

	until checkIfDimensionIndexArrayAreEqual(currentDimensionIndexArray, dimensionSizeArray)

	return AqwamTensorLibrary.construct(newData, newDimensionSizeArray, mode)
	
end

function AqwamTensorLibrary:destroy()

	self.data = nil
	
	self.dimensionSizeArray = nil

	setmetatable(self, nil)

end

return AqwamTensorLibrary
