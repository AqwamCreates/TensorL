local maximumTableLength = 2 ^ 26

local defaultMode = "Row"

local AqwamTensorLibrary = {}

local function checkIfDimensionIndexArraysAreEqual(dimensionIndexArray1, dimensionIndexArray2)

	if (#dimensionIndexArray1 ~= #dimensionIndexArray2) then return false end

	for i, index in ipairs(dimensionIndexArray1) do

		if (index ~= dimensionIndexArray2[i]) then return false end

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

	--[[

	data[subDataIndex] = {}

	data[subDataIndex][subDataDataIndex] = {}
	
	for i = totalSize, 1, 1 do
		
		if ((totalSize % squaredMaximumTableLength) == 0) then
			
			subDataIndex = subDataIndex + 1
			
			subDataDataIndex = 1
			
			data[subDataIndex] = {}
			
		end
		
		if ((totalSize % maximumTableLength) == 0) then
			
			subDataDataIndex = subDataDataIndex + 1

			data[subDataIndex][subDataDataIndex] = {}

		end
		
	end
	
	--]]

	local numberOfDataIndex = math.floor(totalSize / squaredMaximumTableLength)

	local numberOfSubDataIndex = math.ceil((totalSize - (numberOfDataIndex * squaredMaximumTableLength)) / maximumTableLength)

	for i = 1, (numberOfDataIndex - 1), 1 do table.insert(data, table.create(maximumTableLength, table.create(maximumTableLength, true))) end

	local subData = {}

	for i = 1, numberOfSubDataIndex, 1 do table.insert(subData, {}) end

	table.insert(data, subData)

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

local function incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

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

function AqwamTensorLibrary:expandDimensionSizes(targetDimensionSizeArray)

	local currentData = self.data

	local currentDimensionSizeArray = self.dimensionSizeArray

	local mode = self.mode

	if (#currentDimensionSizeArray ~= #targetDimensionSizeArray) then error("The number of dimensions does not match.") end

	for i, size in ipairs(currentDimensionSizeArray) do

		if (size ~= targetDimensionSizeArray[i]) and (size ~= 1) then error("Unable to expand at dimension " .. i .. ".") end

	end

	local targetData

	local newSubTargetDimensionSizeArray

	local oldSubTargetDimensionSizeArray = table.clone(currentDimensionSizeArray)

	local currentDimensionIndexArray = table.create(#currentDimensionSizeArray, 1)

	local dimensionIndexArrayToEndLoop = table.create(#currentDimensionSizeArray, 1)

	local getLinearIndex = getLinearIndexFunctionList[mode]

	for dimension = #currentDimensionSizeArray, 1, -1 do

		local newDimensionSize = targetDimensionSizeArray[dimension]

		local oldDimensionSize = oldSubTargetDimensionSizeArray[dimension]

		newSubTargetDimensionSizeArray = table.clone(oldSubTargetDimensionSizeArray)

		newSubTargetDimensionSizeArray[dimension] = newDimensionSize

		targetData = createEmptyDataFromDimensionSizeArray(newSubTargetDimensionSizeArray)

		repeat

			local subCurrentDimensionIndexArray = table.clone(currentDimensionIndexArray)

			if (oldDimensionSize == 1) and (newDimensionSize >= 2) then

				local currentLinearIndex = getLinearIndex(currentDimensionIndexArray, oldSubTargetDimensionSizeArray)

				local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

				local value = currentData[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex]

				for newDimensionIndex = 1, newDimensionSize, 1 do

					subCurrentDimensionIndexArray[dimension] = newDimensionIndex

					local targetLinearIndex = getLinearIndex(subCurrentDimensionIndexArray, newSubTargetDimensionSizeArray)

					local targetDataIndex, targetSubDataIndex, targetSubSubDataIndex = getDataIndex(targetLinearIndex)

					targetData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] = value

				end

			else

				for newDimensionIndex = 1, newDimensionSize, 1 do

					subCurrentDimensionIndexArray[dimension] = newDimensionIndex

					local currentLinearIndex = getLinearIndex(subCurrentDimensionIndexArray, oldSubTargetDimensionSizeArray)

					local targetLinearIndex = getLinearIndex(subCurrentDimensionIndexArray, newSubTargetDimensionSizeArray)

					local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

					local targetDataIndex, targetSubDataIndex, targetSubSubDataIndex = getDataIndex(targetLinearIndex)

					targetData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] = currentData[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex]

				end

			end

			currentDimensionIndexArray = incrementDimensionIndexArray(currentDimensionIndexArray, oldSubTargetDimensionSizeArray)

		until checkIfDimensionIndexArraysAreEqual(currentDimensionIndexArray, dimensionIndexArrayToEndLoop)

		oldSubTargetDimensionSizeArray = newSubTargetDimensionSizeArray

		currentData = targetData

	end

	return AqwamTensorLibrary.construct(targetData, deepCopyTable(targetDimensionSizeArray), mode)

end

function AqwamTensorLibrary:expandNumberOfDimensions(dimensionSizeToAddArray)

	local currentData = self.data

	local currentDimensionSizeArray = self.dimensionSizeArray

	local mode = self.mode

	local targetDimensionSizeArray = {}

	for i, dimensionSize in ipairs(dimensionSizeToAddArray) do table.insert(targetDimensionSizeArray, dimensionSize) end

	for i, dimensionSize in ipairs(currentDimensionSizeArray) do table.insert(targetDimensionSizeArray, dimensionSize) end

	local targetDimensionIndexArray = table.create(#targetDimensionSizeArray, 1)

	local targetDimensionIndexArrayToEndLoop = table.create(#targetDimensionSizeArray, 1)

	local currentDimensionIndexArray = table.create(#currentDimensionSizeArray, 1)

	local getLinearIndex = getLinearIndexFunctionList[mode]

	local targetData = createEmptyDataFromDimensionSizeArray(targetDimensionSizeArray)

	repeat

		local targetLinearIndex = getLinearIndex(targetDimensionIndexArray, targetDimensionSizeArray)

		local currentLinearIndex = getLinearIndex(currentDimensionIndexArray, currentDimensionSizeArray)

		local targetDataIndex, targetSubDataIndex, targetSubSubDataIndex = getDataIndex(targetLinearIndex)

		local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

		targetData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] = currentData[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex]

		targetDimensionIndexArray = incrementDimensionIndexArray(targetDimensionIndexArray, targetDimensionSizeArray)

		currentDimensionIndexArray = incrementDimensionIndexArray(currentDimensionIndexArray, currentDimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(targetDimensionIndexArray, targetDimensionIndexArrayToEndLoop)

	--[[ They're the same. Don't delete it. Just in case.
	
	local getLinearIndex = getLinearIndexFunctionList[mode]
	
	local targetData
	
	local targetDimensionSizeArray = table.clone(currentDimensionSizeArray)
	
	for i = #dimensionSizeToAddArray, 1, -1 do
		
		table.insert(targetDimensionSizeArray, 1, dimensionSizeToAddArray[i])
		
		targetData = createEmptyDataFromDimensionSizeArray(targetDimensionSizeArray)
		
		local targetDimensionIndexArray = table.create(#targetDimensionSizeArray, 1)
		
		local currentDimensionIndexArray = table.create(#currentDimensionSizeArray, 1)
		
		local targetDimensionIndexArrayToEndLoop = table.create(#targetDimensionSizeArray, 1)
		
		repeat

			local targetLinearIndex = getLinearIndex(targetDimensionIndexArray, targetDimensionSizeArray)

			local currentLinearIndex = getLinearIndex(currentDimensionIndexArray, currentDimensionSizeArray)

			local targetDataIndex, targetSubDataIndex, targetSubSubDataIndex = getDataIndex(targetLinearIndex)

			local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

			targetData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] = currentData[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex]

			targetDimensionIndexArray = incrementDimensionIndexArray(targetDimensionSizeArray, targetDimensionIndexArray)

			currentDimensionIndexArray = incrementDimensionIndexArray(currentDimensionSizeArray, currentDimensionIndexArray)

		until checkIfDimensionIndexArraysAreEqual(targetDimensionIndexArray, targetDimensionIndexArrayToEndLoop)
		
		currentDimensionSizeArray = table.clone(targetDimensionSizeArray)
		
		currentData = targetData
		
	end
	
	--]]

	return AqwamTensorLibrary.construct(targetData, targetDimensionSizeArray, mode)

end

local function getTensorDimensionSizeArrayString(dimensionSizeArray)

	local numberOfDimensions = #dimensionSizeArray

	local tensorDimensionSizeArrayString = "("

	for s, size in ipairs(dimensionSizeArray) do

		tensorDimensionSizeArrayString = tensorDimensionSizeArrayString .. size

		if (s ~= numberOfDimensions) then

			tensorDimensionSizeArrayString = tensorDimensionSizeArrayString .. ", "

		end

	end

	tensorDimensionSizeArrayString = tensorDimensionSizeArrayString .. ")"

	return tensorDimensionSizeArrayString

end

local function onBroadcastError(dimensionSizeArray1, dimensionSizeArray2)

	local tensor1DimensionSizeArrayString = getTensorDimensionSizeArrayString(dimensionSizeArray1)

	local tensor2DimensionSizeArrayString = getTensorDimensionSizeArrayString(dimensionSizeArray2)

	local errorMessage = "Unable to broadcast. " .. "Tensor 1 size: " .. tensor1DimensionSizeArrayString .. " Tensor 2 size: " .. tensor2DimensionSizeArrayString

	error(errorMessage)

end

local function getTheDimensionSizeArrayWithFewestNumberOfDimensionSizeOf1(dimensionSizeArray1, dimensionSizeArray2)

	local dimensionSizeOf1Count1 = 0

	local dimensionSizeOf1Count2 = 0

	for i = 1, #dimensionSizeArray1, 1 do

		if (dimensionSizeArray1[i] == 1) then dimensionSizeOf1Count1 = dimensionSizeOf1Count1 + 1 end

		if (dimensionSizeArray2[i] == 1) then dimensionSizeOf1Count2 = dimensionSizeOf1Count2 + 1 end

	end

	if (dimensionSizeOf1Count1 == 0) then

		return 2

	elseif (dimensionSizeOf1Count2 == 0) then

		return 1

	end

	if (dimensionSizeOf1Count1 > dimensionSizeOf1Count2) then

		return 1

	else

		return 2

	end

end

local function broadcast(tensor1, tensor2, deepCopyOriginalTensor)

	local dimensionSizeArray1 = tensor1:getDimensionSizeArray()

	local dimensionSizeArray2 = tensor2:getDimensionSizeArray()

	if checkIfDimensionIndexArraysAreEqual(dimensionSizeArray1, dimensionSizeArray2) then 

		if (deepCopyOriginalTensor) then

			return deepCopyTable(tensor1), deepCopyTable(tensor2)

		else

			return tensor1, tensor2 

		end

	end

	if (type(tensor1) ~= "table") then 

		tensor1 = AqwamTensorLibrary.new({tensor1})

		dimensionSizeArray1[1] = 1

	end

	if (type(tensor2) ~= "table") then 

		tensor2 = AqwamTensorLibrary.new({tensor2})

		dimensionSizeArray2[1] = 1

	end

	local numberOfDimensions1 = #dimensionSizeArray1 

	local numberOfDimensions2 = #dimensionSizeArray2

	local tensorNumberWithLowestNumberOfDimensions

	if (numberOfDimensions1 == numberOfDimensions2) then -- Currently, if the number of dimensions have the same size, the tensor containing dimension with smaller axis will not expandDimensionSizes. See case when tensor sizes are (5, 3, 6) and (5, 1, 6). So we need to be explicit in our dimensionSizeArrayWithHighestNumberOfDimensions variable.

		tensorNumberWithLowestNumberOfDimensions = getTheDimensionSizeArrayWithFewestNumberOfDimensionSizeOf1(dimensionSizeArray1, dimensionSizeArray2)

	else

		tensorNumberWithLowestNumberOfDimensions = ((numberOfDimensions1 < numberOfDimensions2) and 1) or 2

	end

	local isTensor1HaveLessNumberOfDimensions = (tensorNumberWithLowestNumberOfDimensions == 1)

	local tensorWithLowestNumberOfDimensions = (isTensor1HaveLessNumberOfDimensions and tensor1) or tensor2

	local dimensionSizeArrayWithLowestNumberOfDimensions = (isTensor1HaveLessNumberOfDimensions and dimensionSizeArray1) or dimensionSizeArray2

	local dimensionSizeArrayWithHighestNumberOfDimensions = ((not isTensor1HaveLessNumberOfDimensions) and dimensionSizeArray1) or dimensionSizeArray2

	local lowestNumberOfDimensions = #dimensionSizeArrayWithLowestNumberOfDimensions

	local highestNumberOfDimensions = #dimensionSizeArrayWithHighestNumberOfDimensions

	local numberOfDimensionDifferences = highestNumberOfDimensions - lowestNumberOfDimensions

	local truncatedDimensionSizeArrayWithHighestNumberOfDimensions = table.clone(dimensionSizeArrayWithHighestNumberOfDimensions)

	for i = 1, numberOfDimensionDifferences, 1 do -- We need to remove the extra dimensions from tensor with highest number of dimensions. The values are removed starting from the first so that we can compare the endings.

		table.remove(truncatedDimensionSizeArrayWithHighestNumberOfDimensions, 1)

	end

	for i, dimensionSize in ipairs(dimensionSizeArrayWithLowestNumberOfDimensions) do -- Check if the endings are equal so that we can broadcast one of the tensor. If the dimension size are not equal and neither have dimension size of 1, then we can't broadcast the tensor with the lowest number of dimensions.

		if (dimensionSize ~= truncatedDimensionSizeArrayWithHighestNumberOfDimensions[i]) and (dimensionSize ~= 1) then onBroadcastError(dimensionSizeArray1, dimensionSizeArray2) end

	end

	local dimensionSizeToAddArray = {}

	for i = 1, numberOfDimensionDifferences, 1 do table.insert(dimensionSizeToAddArray, dimensionSizeArrayWithHighestNumberOfDimensions[i]) end -- Get the dimension sizes of the left part of dimension size array.

	local expandedTensor = tensorWithLowestNumberOfDimensions:expandNumberOfDimensions(dimensionSizeToAddArray)

	expandedTensor = expandedTensor:expandDimensionSizes(dimensionSizeArrayWithHighestNumberOfDimensions)

	if (tensorNumberWithLowestNumberOfDimensions == 1) then

		if (deepCopyOriginalTensor) then

			return expandedTensor, deepCopyTable(tensor2)

		else

			return expandedTensor, tensor2 

		end

	else

		if (deepCopyOriginalTensor) then

			return deepCopyTable(tensor1), expandedTensor

		else

			return tensor1, expandedTensor

		end

	end

end

function AqwamTensorLibrary:broadcast(tensor1, tensor2)

	return broadcast(tensor1, tensor2, true)

end


local function applyFunctionUsingOneTensor(functionToApply, tensor)

	local targetData = {}

	for _, subData in ipairs(tensor.data) do 

		local newTargetSubData = {}

		for _, subSubData in ipairs(subData) do 

			local newTargetSubSubData = {}

			for _, value in ipairs(subSubData) do table.insert(newTargetSubSubData, functionToApply(value)) end

			table.insert(newTargetSubData, newTargetSubSubData)

		end

		table.insert(targetData, newTargetSubData)

	end

	return targetData

end

local function applyFunctionUsingTwoTensorsOfSameModes(functionToApply, tensor1, tensor2)

	local targetData = {}

	for i, subData1 in ipairs(tensor1.data) do 

		local subData2 = tensor2[i]

		local newTargetSubData = {}

		for j, subSubData1 in ipairs(subData1) do 

			local subSubData2 = subData2[j]

			local newTargetSubSubData = {}

			for k, value in ipairs(subSubData1) do table.insert(newTargetSubSubData, functionToApply(value, subSubData2[k])) end

			table.insert(newTargetSubData, newTargetSubSubData)

		end

		table.insert(targetData, newTargetSubData)

	end

	return AqwamTensorLibrary.construct(targetData, deepCopyTable(tensor1.dimensionSizeArray), deepCopyTable(tensor1.mode))

end

local function applyFunctionUsingTwoTensorsOfDifferentModes(functionToApply, tensor1, tensor2)

	local currentDimensionSizeArray = tensor1.dimensionSizeArray

	local currentDimensionIndexArray = table.create(#currentDimensionSizeArray, 1)

	local dimensionIndexArrayToEndLoop = table.create(#currentDimensionSizeArray, 1)

	local getLinearIndex1 = getLinearIndexFunctionList[tensor1.mode]

	local getLinearIndex2 = getLinearIndexFunctionList[tensor2.mode] 

	local tensor1Data = tensor1.data

	local tensor2Data = tensor2.data

	local targetData = createEmptyDataFromDimensionSizeArray(currentDimensionSizeArray)

	repeat

		local linearIndex1 = getLinearIndex1(currentDimensionIndexArray)

		local linearIndex2 = getLinearIndex2(currentDimensionIndexArray)

		local dataIndex1, subDataIndex1, subSubDataIndex1 = getDataIndex(linearIndex1)

		local dataIndex2, subDataIndex2, subSubDataIndex2 = getDataIndex(linearIndex2)

		targetData[dataIndex1][subDataIndex1][subSubDataIndex1] = functionToApply(tensor1Data[dataIndex1][subDataIndex1][subSubDataIndex1], tensor2Data[dataIndex2][dataIndex2][dataIndex2])

		currentDimensionIndexArray = incrementDimensionIndexArray(currentDimensionIndexArray, currentDimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(currentDimensionIndexArray, dimensionIndexArrayToEndLoop)

	return AqwamTensorLibrary.construct(targetData, deepCopyTable(tensor1.dimensionSizeArray), deepCopyTable(tensor1.mode))

end

local function applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2, index)

	local tensor1Mode = tensor1.mode

	local tensor2Mode = tensor2.mode

	if (tensor1Mode ~= "Row") and (tensor1Mode ~= "Column") then error("Tensor " .. (index - 1) .. " contains an invalid mode.") end -- Index is subtracted by one because it starts at 2 instead of 1.

	if (tensor2Mode ~= "Row") and (tensor2Mode ~= "Column") then error("Tensor " .. index .. " contains an invalid mode.") end

	if (tensor1Mode == tensor2Mode) then return applyFunctionUsingTwoTensorsOfSameModes(functionToApply, tensor1, tensor2) end

	return applyFunctionUsingTwoTensorsOfDifferentModes(functionToApply, tensor1, tensor2)

end

local function applyFunctionWhenTheFirstValueIsAScalar(functionToApply, scalar, tensor)

	local newData = {}

	for _, subData in ipairs(tensor.data) do 

		local newTargetSubData = {}

		for _, subSubData in ipairs(subData) do 

			local newSubSubData = {}

			for _, value in ipairs(subSubData) do table.insert(newSubSubData, functionToApply(scalar, value)) end

			table.insert(newTargetSubData, newSubSubData)

		end

		table.insert(newData, newTargetSubData)

	end

	return AqwamTensorLibrary.construct(newData, deepCopyTable(tensor.dimensionSizeArray), deepCopyTable(tensor.mode))

end

local function applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, scalar)

	local newData = {}

	for _, subData in ipairs(tensor.data) do 

		local newTargetSubData = {}

		for _, subSubData in ipairs(subData) do 

			local newTargetSubSubData = {}

			for _, value in ipairs(subSubData) do table.insert(newTargetSubSubData, functionToApply(value, scalar)) end

			table.insert(newTargetSubData, newTargetSubSubData)

		end

		table.insert(newData, newTargetSubData)

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

			tensor, otherTensor = broadcast(tensor, otherTensor, false)

			tensor = applyFunctionUsingTwoTensors(functionToApply, tensor, otherTensor, i)

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

	return applyFunctionOnMultipleTensors(function(a, b) return (a + b) end, self, other)

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

	return applyFunctionOnMultipleTensors(function(a, b) return (a * b) end, self, other)

end

function AqwamTensorLibrary:multiply(...)

	local functionToApply = function(a, b) return (a * b) end

	if (self.dimensionSizeArray) then

		return applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		return applyFunctionOnMultipleTensors(functionToApply, ...)

	end

end

function AqwamTensorLibrary:__div(other)

	return applyFunctionOnMultipleTensors(function(a, b) return (a / b) end, self, other)

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

	local currentDimensionSizeArray = self.dimensionSizeArray

	if (#dimensionArray ~= 2) then error("Dimension array must contain 2 dimensions.") end

	local dimension1 = dimensionArray[1]

	local dimension2 = dimensionArray[2]

	local numberOfDimensions = #currentDimensionSizeArray

	if (dimension1 <= 0) then error("The first dimension must be greater than zero.") end

	if (dimension2 <= 0) then error("The second dimension must be greater than zero.") end

	if (dimension1 > numberOfDimensions) then error("The first dimension exceeds the tensor's number of dimensions") end

	if (dimension2 > numberOfDimensions) then error("The second dimension exceeds the tensor's number of dimensions") end

	if (dimension1 == dimension2) then error("The first dimension is equal to the second dimension.") end

	local targetDimensionSizeArray = table.clone(currentDimensionSizeArray)

	targetDimensionSizeArray[dimension1] = currentDimensionSizeArray[dimension2]

	targetDimensionSizeArray[dimension2] = currentDimensionSizeArray[dimension1]

	local getLinearIndex = getLinearIndexFunctionList[self.mode]

	local currentDimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(#currentDimensionSizeArray, 1)

	local currentData = self.data

	local targetData = createEmptyDataFromDimensionSizeArray(targetDimensionSizeArray)

	repeat

		local targetDimensionIndexArray = table.clone(currentDimensionIndexArray)

		targetDimensionIndexArray[dimension1] = currentDimensionIndexArray[dimension2]

		targetDimensionIndexArray[dimension2] = currentDimensionIndexArray[dimension1]

		local currentLinearIndex = getLinearIndex(currentDimensionIndexArray, currentDimensionSizeArray)

		local targetLinearIndex = getLinearIndex(targetDimensionIndexArray, targetDimensionSizeArray)

		local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

		local targetDataIndex, targetSubDataIndex, targetSubSubDataIndex = getDataIndex(targetLinearIndex)

		targetData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] = currentData[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex]

		currentDimensionIndexArray = incrementDimensionIndexArray(currentDimensionIndexArray, currentDimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(currentDimensionIndexArray, dimensionIndexArrayToEndLoop)

	return AqwamTensorLibrary.construct(targetData, targetDimensionSizeArray, self.mode)

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

	local currentData = self.data

	if (not dimension) then return sumFromAllDimensionsFromData(currentData) end

	if (type(dimension) ~= "number") then error("The dimension must be a number.") end

	local currentDimensionSizeArray = self.dimensionSizeArray

	local numberOfDimensions = #currentDimensionSizeArray

	throwErrorIfDimensionIsOutOfBounds(dimension, 1, numberOfDimensions)

	local mode = self.mode
	
	local targetDimensionSizeArray = table.clone(currentDimensionSizeArray)

	targetDimensionSizeArray[dimension] = 1

	local getLinearIndex = getLinearIndexFunctionList[mode]

	local currentDimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(#currentDimensionSizeArray, 1)

	local targetData = createEmptyDataFromDimensionSizeArray(targetDimensionSizeArray)

	repeat

		local targetDimensionIndexArray = table.clone(currentDimensionIndexArray)

		targetDimensionIndexArray[dimension] = 1

		local currentLinearIndex = getLinearIndex(currentDimensionIndexArray, currentDimensionSizeArray)

		local targetLinearIndex = getLinearIndex(targetDimensionIndexArray, targetDimensionSizeArray)

		local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

		local targetDataIndex, targetSubDataIndex, targetSubSubDataIndex = getDataIndex(targetLinearIndex)

		local newDataValue = targetData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] or 0

		targetData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] = newDataValue + currentData[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex]

		currentDimensionIndexArray = incrementDimensionIndexArray(currentDimensionIndexArray, currentDimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(currentDimensionIndexArray, dimensionIndexArrayToEndLoop)

	return AqwamTensorLibrary.construct(targetData, targetDimensionSizeArray, mode)

end

function AqwamTensorLibrary:squeeze(dimension)

	if (type(dimension) ~= "number") then error("The dimension must be a number.") end

	local currentDimensionSizeArray = self.dimensionSizeArray
	
	local numberOfDimensions = #currentDimensionSizeArray
	
	throwErrorIfDimensionIsOutOfBounds(dimension, 1, numberOfDimensions)

	if (currentDimensionSizeArray[dimension] ~= 1) then error("The dimension size at dimension " .. dimension .. " is not equal to 1.") end

	local currentData = self.data

	local mode = self.mode

	local getLinearIndex = getLinearIndexFunctionList[mode]
	
	local targetDimensionSizeArray = table.clone(currentDimensionSizeArray)

	table.remove(targetDimensionSizeArray, dimension)

	local currentDimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(#currentDimensionSizeArray, 1)

	local targetData = createEmptyDataFromDimensionSizeArray(targetDimensionSizeArray)

	repeat

		local targetDimensionIndexArray = table.clone(currentDimensionIndexArray)

		table.remove(targetDimensionIndexArray, dimension)

		local currentLinearIndex = getLinearIndex(currentDimensionIndexArray, currentDimensionSizeArray)

		local targetLinearIndex = getLinearIndex(targetDimensionIndexArray, targetDimensionSizeArray)

		local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

		local targetDataIndex, targetSubDataIndex, targetSubSubDataIndex = getDataIndex(targetLinearIndex)

		local newDataValue = targetData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] or 0

		targetData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] = currentData[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex]

		currentDimensionIndexArray = incrementDimensionIndexArray(currentDimensionIndexArray, currentDimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(currentDimensionIndexArray, dimensionIndexArrayToEndLoop)
	
	return AqwamTensorLibrary.construct(targetData, targetDimensionSizeArray, mode)

end

function AqwamTensorLibrary:mean(dimension)

	local size = (dimension and self:getDimensionSizeArray()[dimension]) or self:getTotalSize()

	local sumTensor = self:sum(dimension)

	local meanTensor = sumTensor:divide(size)

	return meanTensor

end

function AqwamTensorLibrary:standardDeviation(dimension)

	local size = (dimension and self:getDimensionSizeArray()[dimension]) or self:getTotalSize()

	local meanTensor = self:mean(dimension)

	local subtractedTensor = self:subtract(meanTensor)

	local squaredSubractedTensor = subtractedTensor:power(2)

	local summedSquaredSubtractedTensor = squaredSubractedTensor:sum(dimension)

	local squaredStandardDeviationTensor = summedSquaredSubtractedTensor:divide(size)

	local standardDeviationTensor = squaredStandardDeviationTensor:power(0.5)

	return standardDeviationTensor, meanTensor

end

function AqwamTensorLibrary:zScoreNormalization(dimension)

	local standardDeviationTensor, meanTensor = self:standardDeviation(dimension)

	local subtractedTensor = self:subtract(meanTensor)

	local normalizedTensor = subtractedTensor:divide(standardDeviationTensor)

	return normalizedTensor, standardDeviationTensor, meanTensor

end

local function dotProduct(tensor1, tensor2)

	local dimensionSizeArray1 =  tensor1:getDimensionSizeArray()

	local dimensionSizeArray2 =  tensor2:getDimensionSizeArray()

	local numberOfDimensions = #dimensionSizeArray1

	local numberOfDimensionsSubtractedByOne = numberOfDimensions - 1

	if (dimensionSizeArray1[numberOfDimensions] ~= dimensionSizeArray2[numberOfDimensionsSubtractedByOne]) then error("Unable to perform the dot product. The size of second last dimension of first tensor does not equal to the size of the last dimension of second tensor.") end

	for i = 1, (numberOfDimensions - 2), 1  do

		if (dimensionSizeArray1[i] ~= dimensionSizeArray2[i]) then error("Unable to perform the dot product. The size of dimension " .. i .. " of first tensor does not equal to the size of dimension " .. i .. " of second tensor.") end

	end

	local mode1 = tensor1.mode

	local targetDimensionSizeArray = table.clone(dimensionSizeArray1)

	targetDimensionSizeArray[numberOfDimensions] = dimensionSizeArray2[numberOfDimensions]

	local finalDimensionSize = dimensionSizeArray1[numberOfDimensions]

	local getLinearIndex1 = getLinearIndexFunctionList[mode1]

	local getLinearIndex2 = getLinearIndexFunctionList[tensor2.mode]

	local currentDimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	local targetData = createEmptyDataFromDimensionSizeArray(targetDimensionSizeArray)

	repeat

		local currentTensor1DimensionIndexArray = table.clone(currentDimensionIndexArray)

		local currentTensor2DimensionIndexArray = table.clone(currentDimensionIndexArray)

		local sumValue = 0

		for i = 1, finalDimensionSize, 1 do -- Tensor 1 last dimension has the same size to tensor 2 second last dimension. They're also summed together.

			currentTensor1DimensionIndexArray[numberOfDimensions] = i

			currentTensor2DimensionIndexArray[numberOfDimensionsSubtractedByOne] = i

			local linearIndex1 = getLinearIndex1(currentTensor1DimensionIndexArray, dimensionSizeArray1)

			local linearIndex2 = getLinearIndex2(currentTensor2DimensionIndexArray, dimensionSizeArray2)

			local dataIndex1, subDataIndex1, subSubDataIndex1 = getDataIndex(linearIndex1)

			local dataIndex2, subDataIndex2, subSubDataIndex2 = getDataIndex(linearIndex2)

			sumValue = sumValue + (tensor1[dataIndex1][subDataIndex1][subSubDataIndex1] * tensor2[dataIndex2][subDataIndex2][subSubDataIndex2])

		end

		local currentLinearIndex = getLinearIndex1(currentDimensionIndexArray, targetDimensionSizeArray)

		local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

		targetData[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex] = sumValue

		currentDimensionIndexArray = incrementDimensionIndexArray(currentDimensionIndexArray, targetDimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(currentDimensionIndexArray, dimensionIndexArrayToEndLoop)

	return AqwamTensorLibrary.construct(targetData, targetDimensionSizeArray, mode1)

end

function AqwamTensorLibrary:dotProduct(...)

	local tensorArray = {...}

	if (self.dimensionSizeArray) then table.insert(tensorArray, 1, self) end

	local tensor = tensorArray[1]

	for i = 2, #tensorArray, 1 do tensor = dotProduct(tensor, tensorArray[i]) end

	return tensor

end

function AqwamTensorLibrary:switchMode()

	local currentData = self.data

	local currentDimensionSizeArray = self.dimensionSizeArray

	local currentMode = self.mode

	local targetMode = ((currentMode == "Row") and "Column") or "Row"

	local currentDimensionIndexArray = table.create(#currentDimensionSizeArray, 1)

	local dimensionIndexArrayToEndLoop = table.create(#currentDimensionSizeArray, 1)

	local getCurrentLinearIndex = getLinearIndexFunctionList[currentMode]

	local getTargetLinearIndex = getLinearIndexFunctionList[targetMode]

	local targetData = createEmptyDataFromDimensionSizeArray(currentDimensionSizeArray)

	repeat

		local targetLinearIndex = getTargetLinearIndex(currentDimensionIndexArray, currentDimensionSizeArray)

		local currentLinearIndex = getCurrentLinearIndex(currentDimensionIndexArray, currentDimensionSizeArray)

		local targetDataIndex, targetSubDataIndex, targetSubSubDataIndex = getDataIndex(targetLinearIndex)

		local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

		targetData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] = currentData[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex]

		currentDimensionIndexArray = incrementDimensionIndexArray(currentDimensionIndexArray, currentDimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(currentDimensionIndexArray, dimensionIndexArrayToEndLoop)

	return AqwamTensorLibrary.construct(targetData, currentDimensionSizeArray, targetMode)

end

function AqwamTensorLibrary:permute(targetDimensionArray)

	local currentData = self.data

	local currentDimensionSizeArray = self.dimensionSizeArray

	local mode = self.mode

	local numberOfDimensions = #currentDimensionSizeArray

	if (numberOfDimensions ~= #targetDimensionArray) then error("The number of dimensions does not match.") end

	local collectedTargetDimensionArray = {}

	for i, dimension in ipairs(targetDimensionArray) do

		if (dimension > numberOfDimensions) then error("Value of " .. dimension .. " in the target dimension array exceeds the number of dimensions.") end

		if (table.find(collectedTargetDimensionArray, dimension)) then error("Value of " .. dimension .. " in the target dimension array has been added more than once.") end

		table.insert(collectedTargetDimensionArray, dimension)

	end

	local currentDimensionIndexArray = table.create(#currentDimensionSizeArray, 1)

	local dimensionIndexArrayToEndLoop = table.create(#currentDimensionSizeArray, 1)

	local targetDimensionIndexArray = {}

	local targetDimensionSizeArray = {}

	for i, dimension in ipairs(targetDimensionArray) do targetDimensionSizeArray[i] = currentDimensionSizeArray[dimension] end

	local getLinearIndex = getLinearIndexFunctionList[mode]

	local targetData = createEmptyDataFromDimensionSizeArray(currentDimensionSizeArray)

	repeat

		for i, dimension in ipairs(targetDimensionArray) do targetDimensionIndexArray[i] = currentDimensionIndexArray[dimension] end

		local targetLinearIndex = getLinearIndex(targetDimensionIndexArray, targetDimensionSizeArray)

		local currentLinearIndex = getLinearIndex(currentDimensionIndexArray, currentDimensionSizeArray)

		local targetDataIndex, targetSubDataIndex, targetSubSubDataIndex = getDataIndex(targetLinearIndex)

		local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

		targetData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] = currentData[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex]

		currentDimensionIndexArray = incrementDimensionIndexArray(currentDimensionIndexArray, currentDimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(currentDimensionIndexArray, dimensionIndexArrayToEndLoop)

	return AqwamTensorLibrary.construct(targetData, targetDimensionSizeArray, mode)

end

local function getOutOfBoundsIndexArray(array, arrayToBeCheckedForOutOfBounds)

	local outOfBoundsIndexArray = {}

	for i, value in ipairs(arrayToBeCheckedForOutOfBounds) do

		if (value < 1) or (value > array[i]) then table.insert(outOfBoundsIndexArray, i) end

	end

	return outOfBoundsIndexArray

end

function AqwamTensorLibrary:extract(originDimensionIndexArray, targetDimensionIndexArray)

	local currentData = self.data

	local currentDimensionSizeArray = self.dimensionSizeArray

	local mode = self.mode

	local numberOfDimensions = #currentDimensionSizeArray

	if (numberOfDimensions ~= #originDimensionIndexArray) then error("Invalid origin dimension index array.") end

	if (numberOfDimensions ~= #targetDimensionIndexArray) then error("Invalid target dimension index array.") end

	local outOfBoundsOriginIndexArray = getOutOfBoundsIndexArray(currentDimensionSizeArray, originDimensionIndexArray)

	local outOfBoundsTargetIndexArray = getOutOfBoundsIndexArray(currentDimensionSizeArray, targetDimensionIndexArray)

	local outOfBoundsOriginIndexArraySize = #outOfBoundsOriginIndexArray

	local outOfBoundsTargetIndexArraySize = #outOfBoundsTargetIndexArray

	if (outOfBoundsOriginIndexArraySize > 0) then

		local errorString = "Attempting to set an origin dimension index that is out of bounds for dimension at "

		for i, index in ipairs(outOfBoundsOriginIndexArray) do

			errorString = errorString .. index

			if (i < outOfBoundsOriginIndexArraySize) then errorString = errorString .. ", " end

		end

		errorString = errorString .. "."

		error(errorString)

	end

	if (outOfBoundsTargetIndexArraySize > 0) then

		local errorString = "Attempting to set an target dimension index that is out of bounds for dimension at "

		for i, index in ipairs(outOfBoundsTargetIndexArray) do

			errorString = errorString .. index

			if (i < outOfBoundsTargetIndexArraySize) then errorString = errorString .. ", " end

		end

		errorString = errorString .. "."

		error(errorString)

	end

	local currentDimensionIndexArray = table.create(#currentDimensionSizeArray, 1)

	local dimensionIndexArrayToEndLoop = table.create(#currentDimensionSizeArray, 1)

	local targetDimensionIndexArray = {}

	local targetDimensionSizeArray = {}

	for i, dimension in ipairs(targetDimensionArray) do targetDimensionSizeArray[i] = currentDimensionSizeArray[dimension] end

	local getLinearIndex = getLinearIndexFunctionList[mode]

	local targetData = createEmptyDataFromDimensionSizeArray(currentDimensionSizeArray)

	repeat

		for i, dimension in ipairs(targetDimensionArray) do targetDimensionIndexArray[i] = currentDimensionIndexArray[dimension] end

		local targetLinearIndex = getLinearIndex(targetDimensionIndexArray, targetDimensionSizeArray)

		local currentLinearIndex = getLinearIndex(currentDimensionIndexArray, currentDimensionSizeArray)

		local targetDataIndex, targetSubDataIndex, targetSubSubDataIndex = getDataIndex(targetLinearIndex)

		local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

		targetData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] = currentData[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex]

		currentDimensionIndexArray = incrementDimensionIndexArray(currentDimensionIndexArray, currentDimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(currentDimensionIndexArray, dimensionIndexArrayToEndLoop)

end

function AqwamTensorLibrary:copy()

	return AqwamTensorLibrary.construct(deepCopyTable(self.data), deepCopyTable(self.dimensionSizeArray), deepCopyTable(self.mode))

end

function AqwamTensorLibrary:destroy()

	self.data = nil

	self.dimensionSizeArray = nil

	setmetatable(self, nil)

end

return AqwamTensorLibrary
