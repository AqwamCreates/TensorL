--[[

	--------------------------------------------------------------------

	Version 0.9.0

	Aqwam's Tensor Library (TensorL)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	By using or possesing any copies of this library, you agree to our terms and conditions at:
	
	https://github.com/AqwamCreates/TensorL/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

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
	
	local nextDimension = currentDimension + 1

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do dataTableTableIndex, dataTableIndex, currentLinearIndex = convertTensorToData(tensor[i], dimensionSizeArray, numberOfDimensions, nextDimension, targetData, dataTableTableIndex, dataTableIndex, currentLinearIndex) end

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
	
	local nextDimension = currentDimension + 1

	if (currentDimension < numberOfDimensions) then

		for i = 1, dimensionSizeArray[currentDimension], 1 do dataTableTableIndex, dataTableIndex, currentLinearIndex = setValueFromFunctionToData(functionToApply, dimensionSizeArray, numberOfDimensions, nextDimension, targetData, dataTableTableIndex, dataTableIndex, currentLinearIndex) end

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

	local squaredMaximumTableLength = math.pow(maximumTableLength, 2) -- The maximum table length is squared because of the nested nature of the data.

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
	
	if (minimumValue) and (maximumValue) then

		if (minimumValue >= maximumValue) then error("The minimum value cannot be greater than or equal to the maximum value.") end

	elseif (not minimumValue) and (maximumValue) then

		if (maximumValue <= 0) then error("The maximum value cannot be less than or equal to zero.") end

	elseif (minimumValue) and (not maximumValue) then

		if (minimumValue >= 0) then error("The minimum value cannot be greater than or equal to zero.") end

	end

	local self = setmetatable({}, AqwamTensorLibrary)

	local data = createEmptyDataFromDimensionSizeArray(dimensionSizeArray)

	local functionToApply = function()

		if (minimumValue) and (maximumValue) then

			return minimumValue + (math.random() * (maximumValue - minimumValue))
			
		elseif (not minimumValue) and (maximumValue) then

			return math.random() * maximumValue

		elseif (minimumValue) and (not maximumValue) then

			return math.random() * minimumValue

		elseif (not minimumValue) and (not maximumValue) then

			return (math.random() * 2) - 1

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
	
	local numberOfDimensions = #currentDimensionSizeArray

	if (numberOfDimensions ~= #targetDimensionSizeArray) then error("The number of dimensions does not match.") end

	for i, size in ipairs(currentDimensionSizeArray) do

		if (size ~= targetDimensionSizeArray[i]) and (size ~= 1) then error("Unable to expand at dimension " .. i .. ".") end

	end

	local targetData

	local newSubTargetDimensionSizeArray

	local oldSubTargetDimensionSizeArray = table.clone(currentDimensionSizeArray)

	local currentDimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

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
	
	local targetNumberOfDimensions = #targetDimensionSizeArray

	local targetDimensionIndexArray = table.create(targetNumberOfDimensions, 1)

	local targetDimensionIndexArrayToEndLoop = table.create(targetNumberOfDimensions, 1)

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

--[[

local function broadcast(tensor1, tensor2, deepCopyOriginalTensor) -- Single tensor broadcast.

	local dimensionSizeArray1 = getDimensionSizeArray(tensor1)

	local dimensionSizeArray2 = getDimensionSizeArray(tensor2)

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

--]]

local function broadcast(tensor1, tensor2, deepCopyOriginalTensor) -- Dual tensor broadcast.

	local dimensionSizeArray1 = getDimensionSizeArray(tensor1)

	local dimensionSizeArray2 = getDimensionSizeArray(tensor2)

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

	local tensorWithHighestNumberOfDimensions = (not isTensor1HaveLessNumberOfDimensions and tensor1) or tensor2

	local dimensionSizeArrayWithLowestNumberOfDimensions = (isTensor1HaveLessNumberOfDimensions and dimensionSizeArray1) or dimensionSizeArray2

	local dimensionSizeArrayWithHighestNumberOfDimensions = ((not isTensor1HaveLessNumberOfDimensions) and dimensionSizeArray1) or dimensionSizeArray2

	local lowestNumberOfDimensions = #dimensionSizeArrayWithLowestNumberOfDimensions

	local highestNumberOfDimensions = #dimensionSizeArrayWithHighestNumberOfDimensions

	local numberOfDimensionDifferences = highestNumberOfDimensions - lowestNumberOfDimensions

	local truncatedDimensionSizeArrayWithHighestNumberOfDimensions = table.clone(dimensionSizeArrayWithHighestNumberOfDimensions)

	for i = 1, numberOfDimensionDifferences, 1 do -- We need to remove the extra dimensions from tensor with highest number of dimensions. The values are removed starting from the first so that we can compare the endings.

		table.remove(truncatedDimensionSizeArrayWithHighestNumberOfDimensions, 1)

	end

	for i, dimensionSize1 in ipairs(dimensionSizeArrayWithLowestNumberOfDimensions) do -- Check if the endings are equal so that we can broadcast one of the tensor. If the dimension size are not equal and neither have dimension size of 1, then we can't broadcast the tensor with the lowest number of dimensions.

		local dimensionSize2 = truncatedDimensionSizeArrayWithHighestNumberOfDimensions[i]

		if (dimensionSize1 ~= dimensionSize2) and (dimensionSize1 ~= 1) and (dimensionSize2 ~= 1) then onBroadcastError(dimensionSizeArray1, dimensionSizeArray2) end

	end

	local dimensionSizeToAddArray = {}

	for i = 1, numberOfDimensionDifferences, 1 do table.insert(dimensionSizeToAddArray, dimensionSizeArrayWithHighestNumberOfDimensions[i]) end -- Get the dimension sizes of the left part of dimension size array.

	local expandedDimensionSizeArrayForLowestNumberOfDimensions = table.clone(dimensionSizeToAddArray)

	for i = 1, lowestNumberOfDimensions, 1 do table.insert(expandedDimensionSizeArrayForLowestNumberOfDimensions, dimensionSizeArrayWithLowestNumberOfDimensions[i]) end

	local targetDimensionSizeArray = {}

	for i = 1, numberOfDimensionDifferences, 1 do table.insert(targetDimensionSizeArray, dimensionSizeArrayWithHighestNumberOfDimensions[i]) end

	for i = 1, lowestNumberOfDimensions, 1 do targetDimensionSizeArray[i + numberOfDimensionDifferences] = math.max(truncatedDimensionSizeArrayWithHighestNumberOfDimensions[i], dimensionSizeArrayWithLowestNumberOfDimensions[i]) end

	local expandedTensorForTheTensorWithLowestNumberOfDimensions = tensorWithLowestNumberOfDimensions:expandNumberOfDimensions(dimensionSizeToAddArray)

	expandedTensorForTheTensorWithLowestNumberOfDimensions = expandedTensorForTheTensorWithLowestNumberOfDimensions:expandDimensionSizes(targetDimensionSizeArray)

	local expandedTensorForTheTensorWithHighestNumberOfDimensions = tensorWithHighestNumberOfDimensions:expandDimensionSizes(targetDimensionSizeArray)

	if (tensorNumberWithLowestNumberOfDimensions == 1) then

		return expandedTensorForTheTensorWithLowestNumberOfDimensions, expandedTensorForTheTensorWithHighestNumberOfDimensions

	else

		return expandedTensorForTheTensorWithHighestNumberOfDimensions, expandedTensorForTheTensorWithLowestNumberOfDimensions

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

local function applyFunction(functionToApply, ...)
	
	local tensorArray = {...}

	local targetData = {}
	
	local tensor1 = tensorArray[1]

	for i, subData1 in ipairs(tensor1.data) do 

		local newTargetSubData = {}

		for j, subSubData1 in ipairs(subData1) do 

			local newTargetSubSubData = {}

			for k, value in ipairs(subSubData1) do
				
				local argumentArray = {value}
				
				for t = 2, #tensorArray, 1 do table.insert(argumentArray, tensorArray[t].data[i][j][k]) end
				
				table.insert(newTargetSubSubData, functionToApply(table.unpack(argumentArray)))
				
			end

			table.insert(newTargetSubData, newTargetSubSubData)

		end

		table.insert(targetData, newTargetSubData)

	end

	return AqwamTensorLibrary.construct(targetData, deepCopyTable(tensor1.dimensionSizeArray), deepCopyTable(tensor1.mode))

end

function AqwamTensorLibrary:applyFunction(functionToApply, ...)

	local tensorArray = {...}

	if (self.data) then table.insert(tensorArray, 1, self) end
	
	local mode = tensorArray[1].mode
	
	for i = 2, #tensorArray, 1 do if (mode ~= tensorArray[i].mode) then error("Tensor " .. i .. " has an incompatible mode.") end end

	local doAllTensorsHaveTheSameDimensionSizeArray

	--[[
		
		A single sweep is not enough to make sure that all tensors have the same dimension size arrays. So, we need to do it multiple times.
		
		Here's an example where the tensors' dimension size array will not match the others in a single sweep: {2, 3, 1}, {1,3}, {5, 1, 1, 1}. 
		
		The first dimension size array needs to match with the third dimension size array, but can only look at the second dimension size array. 
		
		So, we need to propagate the third dimension size array to the nearby dimension size array so that it reaches the first dimension size array. 
		
		In this case, it would be the second dimension size array.
		
	--]]

	repeat

		doAllTensorsHaveTheSameDimensionSizeArray = true

		for i = 1, (#tensorArray - 1), 1 do

			local tensor1 = tensorArray[i]

			local tensor2 = tensorArray[i + 1]

			local dimensionSizeArray1 = tensor1:getDimensionSizeArray()

			local dimensionSizeArray2 = tensor2:getDimensionSizeArray()

			if (not checkIfDimensionIndexArraysAreEqual(dimensionSizeArray1, dimensionSizeArray2)) then doAllTensorsHaveTheSameDimensionSizeArray = false end

			tensorArray[i], tensorArray[i + 1] = broadcast(tensor1, tensor2, false)

		end

	until (doAllTensorsHaveTheSameDimensionSizeArray)

	return applyFunction(functionToApply, table.unpack(tensorArray))

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

function AqwamTensorLibrary:logarithm(...)

	local functionToApply = math.log

	if (self.dimensionSizeArray) then

		return applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		return applyFunctionOnMultipleTensors(functionToApply, ...)

	end

end

function AqwamTensorLibrary:exponent(...)

	local functionToApply = math.exp

	if (self.dimensionSizeArray) then

		return applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		return applyFunctionOnMultipleTensors(functionToApply, ...)

	end

end

function AqwamTensorLibrary:power(...)

	local functionToApply = math.power

	if (self.dimensionSizeArray) then

		return applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		return applyFunctionOnMultipleTensors(functionToApply, ...)

	end

end

function AqwamTensorLibrary:__pow(other)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a ^ b) end, self, other)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:__mod(other)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a % b) end, self, other)

	return AqwamTensorLibrary.new(resultTensor)

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

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

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

	local varianceTensor = summedSquaredSubtractedTensor:divide(size)

	local standardDeviationTensor = varianceTensor:power(0.5)

	return standardDeviationTensor, varianceTensor, meanTensor

end

function AqwamTensorLibrary:zScoreNormalization(dimension)

	local standardDeviationTensor, varianceTensor, meanTensor = self:standardDeviation(dimension)

	local subtractedTensor = self:subtract(meanTensor)

	local normalizedTensor = subtractedTensor:divide(standardDeviationTensor)

	return normalizedTensor, standardDeviationTensor, varianceTensor, meanTensor

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

local function expandedDotProduct(tensor1, tensor2)

	local dimensionSizeArray1 = tensor1:getDimensionSizeArray()

	local dimensionSizeArray2 = tensor2:getDimensionSizeArray()

	local numberOfDimensions1 = #dimensionSizeArray1

	local numberOfDimensions2 = #dimensionSizeArray2

	local highestNumberOfDimensions = math.max(numberOfDimensions1, numberOfDimensions2)

	local numberOfDimensionsOffset1 = highestNumberOfDimensions - numberOfDimensions1

	local numberOfDimensionsOffset2 = highestNumberOfDimensions - numberOfDimensions2

	local expandedTensor1

	local expandedTensor2

	if (numberOfDimensionsOffset1 ~= 0) then

		local dimensionSizeToAddArray = {}

		for i = 1, numberOfDimensionsOffset1, 1 do table.insert(dimensionSizeToAddArray, dimensionSizeArray2[i]) end

		expandedTensor1 = tensor1:expandNumberOfDimensions(dimensionSizeToAddArray)

	else

		expandedTensor1 = tensor1

	end

	if (numberOfDimensionsOffset2 ~= 0) then

		local dimensionSizeToAddArray = {}

		for i = 1, numberOfDimensionsOffset2, 1 do table.insert(dimensionSizeToAddArray, dimensionSizeArray1[i]) end

		expandedTensor2 = tensor2:expandNumberOfDimensions(dimensionSizeToAddArray)

	else

		expandedTensor2 = tensor2

	end

	return dotProduct(expandedTensor1, expandedTensor2)

end

function AqwamTensorLibrary:dotProduct(...)

	local tensorArray = {...}

	if (self.dimensionSizeArray) then table.insert(tensorArray, 1, self) end

	local tensor = tensorArray[1]

	for i = 2, #tensorArray, 1 do tensor = expandedDotProduct(tensor, tensorArray[i]) end

	return tensor

end

function AqwamTensorLibrary:switchMode()

	local currentData = self.data

	local dimensionSizeArray = self.dimensionSizeArray

	local currentMode = self.mode

	local targetMode = ((currentMode == "Row") and "Column") or "Row"

	local currentDimensionIndexArray = table.create(#dimensionSizeArray, 1)

	local dimensionIndexArrayToEndLoop = table.create(#dimensionSizeArray, 1)

	local getCurrentLinearIndex = getLinearIndexFunctionList[currentMode]

	local getTargetLinearIndex = getLinearIndexFunctionList[targetMode]

	local targetData = createEmptyDataFromDimensionSizeArray(dimensionSizeArray)

	repeat

		local targetLinearIndex = getTargetLinearIndex(currentDimensionIndexArray, dimensionSizeArray)

		local currentLinearIndex = getCurrentLinearIndex(currentDimensionIndexArray, dimensionSizeArray)

		local targetDataIndex, targetSubDataIndex, targetSubSubDataIndex = getDataIndex(targetLinearIndex)

		local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

		targetData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] = currentData[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex]

		currentDimensionIndexArray = incrementDimensionIndexArray(currentDimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(currentDimensionIndexArray, dimensionIndexArrayToEndLoop)

	return AqwamTensorLibrary.construct(targetData, table.clone(dimensionSizeArray), targetMode)

end

function AqwamTensorLibrary:permute(dimensionArray)

	local currentData = self.data

	local dimensionSizeArray = self.dimensionSizeArray

	local mode = self.mode

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensions ~= #dimensionArray) then error("The number of dimensions does not match.") end

	local collectedDimensionArray = {}

	for i, dimension in ipairs(dimensionArray) do

		if (dimension > numberOfDimensions) then error("Value of " .. dimension .. " in the target dimension array exceeds the number of dimensions.") end

		if (table.find(collectedDimensionArray, dimension)) then error("Value of " .. dimension .. " in the target dimension array has been added more than once.") end

		table.insert(collectedDimensionArray, dimension)

	end

	local currentDimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	local targetDimensionIndexArray = {}

	local targetDimensionSizeArray = {}

	for i, dimension in ipairs(dimensionArray) do targetDimensionSizeArray[i] = dimensionSizeArray[dimension] end

	local getLinearIndex = getLinearIndexFunctionList[mode]

	local targetData = createEmptyDataFromDimensionSizeArray(dimensionSizeArray)

	repeat

		for i, dimension in ipairs(dimensionArray) do targetDimensionIndexArray[i] = currentDimensionIndexArray[dimension] end

		local targetLinearIndex = getLinearIndex(targetDimensionIndexArray, targetDimensionSizeArray)

		local currentLinearIndex = getLinearIndex(currentDimensionIndexArray, dimensionSizeArray)

		local targetDataIndex, targetSubDataIndex, targetSubSubDataIndex = getDataIndex(targetLinearIndex)

		local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

		targetData[targetDataIndex][targetSubDataIndex][targetSubSubDataIndex] = currentData[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex]

		currentDimensionIndexArray = incrementDimensionIndexArray(currentDimensionIndexArray, dimensionSizeArray)

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

local function checkIfDimensionIndexArrayIsWithinBounds(dimensionIndexArray, isDimensionIndexArrayDirectionSwappedArray, lowerBoundDimensionIndexArray, upperBoundDimensionIndexArray)
	
	for i = #dimensionIndexArray, 1, -1 do
		
		local dimensionIndex = dimensionIndexArray[i]
		
		if (isDimensionIndexArrayDirectionSwappedArray[i]) then
			
			if (dimensionIndex > lowerBoundDimensionIndexArray[i]) or (dimensionIndex < upperBoundDimensionIndexArray[i]) then return false end
			
		else
			
			if (dimensionIndex < lowerBoundDimensionIndexArray[i]) or (dimensionIndex > upperBoundDimensionIndexArray[i]) then return false end
			
		end
		
	end
	
	return true
	
end

local function islowerBoundValueGreaterThanUpperBoundValueInDimensionIndexArray(lowerBoundDimensionIndexArray, upperBoundDimensionIndexArray)
	
	local booleanArray = {}
	
	for i, lowerBoundValue in ipairs(lowerBoundDimensionIndexArray) do booleanArray[i] = (lowerBoundValue > upperBoundDimensionIndexArray[i]) end
	
	return booleanArray
	
end

function AqwamTensorLibrary:extract(originDimensionIndexArray, targetDimensionIndexArray)

	local currentData = self.data

	local dimensionSizeArray = self.dimensionSizeArray

	local mode = self.mode

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensions ~= #originDimensionIndexArray) then error("Invalid origin dimension index array.") end

	if (numberOfDimensions ~= #targetDimensionIndexArray) then error("Invalid target dimension index array.") end

	local outOfBoundsOriginIndexArray = getOutOfBoundsIndexArray(dimensionSizeArray, originDimensionIndexArray)

	local outOfBoundsTargetIndexArray = getOutOfBoundsIndexArray(dimensionSizeArray, targetDimensionIndexArray)

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
	
	local isDimensionIndexArrayDirectionSwappedArray = islowerBoundValueGreaterThanUpperBoundValueInDimensionIndexArray(originDimensionIndexArray, targetDimensionIndexArray)

	local currentDimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	local newDimensionIndexArray = table.create(numberOfDimensions, 1)

	local newDimensionSizeArray = {}

	for i, targetDimensionIndex in ipairs(targetDimensionIndexArray) do newDimensionSizeArray[i] = targetDimensionIndex - originDimensionIndexArray[i] end
	
	for i, dimensionSize in ipairs(newDimensionSizeArray) do newDimensionSizeArray[i] = math.abs(dimensionSize) end

	local getLinearIndex = getLinearIndexFunctionList[mode]

	local newData = createEmptyDataFromDimensionSizeArray(newDimensionSizeArray)

	repeat
		
		if checkIfDimensionIndexArrayIsWithinBounds(currentDimensionIndexArray, isDimensionIndexArrayDirectionSwappedArray, originDimensionIndexArray, targetDimensionIndexArray) then
			
			local copiedNewDimensionIndexArray = table.clone(newDimensionIndexArray)
			
			for i, boolean in ipairs(isDimensionIndexArrayDirectionSwappedArray) do
				
				if (boolean) then copiedNewDimensionIndexArray[i] = (newDimensionSizeArray[i] - copiedNewDimensionIndexArray[i]) + 1 end
				
			end
			
			local newLinearIndex = getLinearIndex(copiedNewDimensionIndexArray, newDimensionSizeArray)

			local currentLinearIndex = getLinearIndex(currentDimensionIndexArray, dimensionSizeArray)

			local newDataIndex, newSubDataIndex, newSubSubDataIndex = getDataIndex(newLinearIndex)

			local currentDataIndex, currentSubDataIndex, currentSubSubDataIndex = getDataIndex(currentLinearIndex)

			newData[newDataIndex][newSubDataIndex][newSubSubDataIndex] = currentData[currentDataIndex][currentSubDataIndex][currentSubSubDataIndex]
			
			newDimensionIndexArray = incrementDimensionIndexArray(newDimensionIndexArray, newDimensionSizeArray)
			
		end

		currentDimensionIndexArray = incrementDimensionIndexArray(currentDimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(currentDimensionIndexArray, dimensionIndexArrayToEndLoop)
	
	return AqwamTensorLibrary.construct(newData, newDimensionSizeArray, mode)

end

function AqwamTensorLibrary:findMaximumValue()
	
	local highestValue = -math.huge
	
	for _, subData in ipairs(self.data) do 

		for _, subSubData in ipairs(subData) do 

			for _, value in ipairs(subSubData) do highestValue = math.max(highestValue, value) end

		end

	end
	
	return highestValue
	
end

function AqwamTensorLibrary:findMinimumValue()

	local lowestValue = math.huge

	for _, subData in ipairs(self.data) do 

		for _, subSubData in ipairs(subData) do 

			for _, value in ipairs(subSubData) do lowestValue = math.min(lowestValue, value) end

		end

	end

	return lowestValue

end

function AqwamTensorLibrary:findMaximumValueDimensionIndexArray()

	local data = self.data

	local dimensionSizeArray = self.dimensionSizeArray

	local numberOfDimensions = #dimensionSizeArray

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	local highestValueDimensionIndexArray

	local highestValue = -math.huge
	
	local getLinearIndex = getLinearIndexFunctionList[self.mode]

	repeat

		local linearIndex = getLinearIndex(dimensionIndexArray, dimensionSizeArray)

		local dataIndex, subDataIndex, subSubDataIndex = getDataIndex(linearIndex)

		local value = data[dataIndex][subDataIndex][subSubDataIndex]

		if (value > highestValue) then

			highestValueDimensionIndexArray = table.clone(dimensionIndexArray)

			highestValue = value

		end

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return highestValueDimensionIndexArray, highestValue

end

function AqwamTensorLibrary:findMinimumValueDimensionIndexArray()
	
	local data = self.data
	
	local dimensionSizeArray = self.dimensionSizeArray

	local numberOfDimensions = #dimensionSizeArray

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	local lowestValueDimensionIndexArray

	local lowestValue = math.huge
	
	local getLinearIndex = getLinearIndexFunctionList[self.mode]

	repeat

		local linearIndex = getLinearIndex(dimensionIndexArray, dimensionSizeArray)
		
		local dataIndex, subDataIndex, subSubDataIndex = getDataIndex(linearIndex)
		
		local value = data[dataIndex][subDataIndex][subSubDataIndex]

		if (value < lowestValue) then

			lowestValueDimensionIndexArray = table.clone(dimensionIndexArray)

			lowestValue = value

		end

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return lowestValueDimensionIndexArray, lowestValue

end

function AqwamTensorLibrary:copy()

	return AqwamTensorLibrary.construct(deepCopyTable(self.data), deepCopyTable(self.dimensionSizeArray), deepCopyTable(self.mode))

end

function AqwamTensorLibrary:destroy()

	self.data = nil

	self.dimensionSizeArray = nil

	setmetatable(self, nil)
	
	self = nil

end

return AqwamTensorLibrary
