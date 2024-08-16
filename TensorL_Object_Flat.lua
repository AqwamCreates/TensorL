--[[

	--------------------------------------------------------------------

	Version 0.6.0

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

local AqwamTensorLibrary = {}

local function checkIfItHasSameDimensionSizeArray(dimensionSizeArray, targetDimensionSizeArray)

	if (#dimensionSizeArray ~= #targetDimensionSizeArray) then return false end

	for i, size in ipairs(dimensionSizeArray) do

		if (size ~= targetDimensionSizeArray[i]) then return false end

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

local function removeFirstValueFromArray(array)

	local newArray = {}

	for i = 2, #array, 1 do table.insert(newArray, array[i]) end

	return newArray

end

local function removeLastValueFromArray(array)

	local newArray = table.clone(array)

	table.remove(newArray, #newArray)

	return newArray

end

local function getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

	local totalSize = 1

	for _, value in ipairs(dimensionSizeArray) do totalSize = value * totalSize end

	return totalSize

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

function AqwamTensorLibrary:getDimensionSizeArray()

	local dimensionSizeArray = {}

	getDimensionSizeArray(self, dimensionSizeArray)

	return dimensionSizeArray

end

local function expand(tensor, dimensionSizeArray, targetDimensionSizeArray)

	-- Does not do the same thing with inefficient expand function. This one expand at the lowest dimension first and then the parent dimension will make copy of this.

	local resultTensor

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensions >= 2) then

		resultTensor = {}

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		local remainingTargetDimensionSizeArray = removeFirstValueFromArray(targetDimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do resultTensor[i] = expand(tensor[i], remainingDimensionSizeArray, remainingTargetDimensionSizeArray) end

	else

		resultTensor = deepCopyTable(tensor)  -- If the "(numberOfDimensions > 1)" from the first "if" statement does not run, it will return the original tensor. So we need to deep copy it.

	end

	local updatedDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(resultTensor) -- Need to call this again because we may have modified the tensor below it, thus changing the dimension size array.

	local dimensionSize = updatedDimensionSizeArray[1]

	local targetDimensionSize = targetDimensionSizeArray[1]

	local hasSameDimensionSize = (dimensionSize == targetDimensionSize)

	local canDimensionBeExpanded = (dimensionSize == 1)

	if (numberOfDimensions >= 1) and (not hasSameDimensionSize) and (canDimensionBeExpanded) then 

		local subTensor = resultTensor[1]

		for i = 1, targetDimensionSize, 1 do resultTensor[i] = deepCopyTable(subTensor) end

	elseif (not hasSameDimensionSize) and (not canDimensionBeExpanded) then

		error("Unable to expand.")

	end

	return resultTensor

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

	if (dimensionSizeOf1Count1 < dimensionSizeOf1Count2) then

		return 1

	else

		return 2

	end

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

function AqwamTensorLibrary:expand(targetDimensionSizeArray)

	local dimensionSizeArray = self:getDimensionSizeArray()

	if checkIfItHasSameDimensionSizeArray(dimensionSizeArray, targetDimensionSizeArray) then return deepCopyTable(self) end -- Do not remove this code even if the code below is related or function similar to this code. You will spend so much time fixing it if you forget that you have removed it.

	local resultTensor = expand(self, dimensionSizeArray, targetDimensionSizeArray)

	return AqwamTensorLibrary.new(resultTensor)

end

local function increaseNumberOfDimensions(tensor, dimensionSizeToAddArray)

	local resultTensor = {}

	local numberOfDimensionsToAdd = #dimensionSizeToAddArray

	if (numberOfDimensionsToAdd > 1) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeToAddArray)

		for i = 1, dimensionSizeToAddArray[1], 1 do resultTensor[i] = increaseNumberOfDimensions(tensor, remainingDimensionSizeArray) end

	elseif (numberOfDimensionsToAdd == 1) then

		for i = 1, dimensionSizeToAddArray[1], 1 do resultTensor[i] = deepCopyTable(tensor) end

	else

		resultTensor = tensor

	end

	return resultTensor

end

function AqwamTensorLibrary:increaseNumberOfDimensions(dimensionSizeToAddArray)

	local resultTensor = increaseNumberOfDimensions(self, dimensionSizeToAddArray)

	return AqwamTensorLibrary.new(resultTensor)

end

local function broadcast(tensor1, tensor2, deepCopyOriginalTensor)

	local dimensionSizeArray1 = AqwamTensorLibrary:getDimensionSizeArray(tensor1)

	local dimensionSizeArray2 = AqwamTensorLibrary:getDimensionSizeArray(tensor2)

	if checkIfItHasSameDimensionSizeArray(dimensionSizeArray1, dimensionSizeArray2) then 

		if (deepCopyOriginalTensor) then

			return deepCopyTable(tensor1), deepCopyTable(tensor2)

		else

			return tensor1, tensor2 

		end

	end

	local numberOfDimensions1 = #dimensionSizeArray1 

	local numberOfDimensions2 = #dimensionSizeArray2

	local tensorNumberWithLowestNumberOfDimensions

	if (numberOfDimensions1 == numberOfDimensions2) then -- Currently, if the number of dimensions have the same size, the tensor containing dimension with smaller axis will not expand. See case when tensor sizes are (5, 3, 6) and (5, 1, 6). So we need to be explicit in our dimensionSizeArrayWithHighestNumberOfDimensions variable.

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

	local expandedTensor = AqwamTensorLibrary:increaseNumberOfDimensions(tensorWithLowestNumberOfDimensions, dimensionSizeToAddArray)

	expandedTensor = AqwamTensorLibrary:expand(expandedTensor, dimensionSizeArrayWithHighestNumberOfDimensions)

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

	local tensor1Value, tensor2Value = broadcast(tensor1.tensor, tensor2.tensor, true)

	return AqwamTensorLibrary.new(tensor1Value),  AqwamTensorLibrary.new(tensor2Value)

end

local function applyFunctionUsingOneTensor(functionToApply, tensor) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local resultTensor = {}

	for i = 1, #tensor, 1 do resultTensor[i] = functionToApply(tensor[i]) end

	return resultTensor

end

local function applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local resultTensor = {}

	for i = 1, #tensor1, 1 do resultTensor[i] = functionToApply(tensor1[i], tensor2[i]) end

	return resultTensor

end

local function applyFunctionWhenTheFirstValueIsAScalar(functionToApply, scalar, tensor) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local resultTensor = {}

	for i = 1, #tensor, 1 do resultTensor[i] = functionToApply(scalar, tensor[i]) end

	return resultTensor

end

local function applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, scalar) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	local resultTensor = {}

	for i = 1, #tensor, 1 do resultTensor[i] = functionToApply(tensor[i], scalar) end

	return resultTensor

end

local function applyFunctionOnMultipleTensors(functionToApply, ...)

	local tensorArray = {...}

	local numberOfTensors = #tensorArray

	local tensor = tensorArray[1]

	if (numberOfTensors == 1) then 

		local dimensionSizeArray = getDimensionSizeArray(tensor)

		if (type(tensor) == "table") then

			return applyFunctionUsingOneTensor(functionToApply, tensor, dimensionSizeArray)

		else

			return functionToApply(tensor, dimensionSizeArray)

		end

	end

	for i = 2, numberOfTensors, 1 do

		local otherTensor = tensorArray[i]

		local isFirstValueATensor = (type(tensor) == "table")

		local isSecondValueATensor = (type(otherTensor) == "table")

		if (isFirstValueATensor) and (isSecondValueATensor) then

			tensor, otherTensor = broadcast(tensor, otherTensor, false)

			local dimensionSizeArray = getDimensionSizeArray(tensor)

			tensor = applyFunctionUsingTwoTensors(functionToApply, tensor, otherTensor, dimensionSizeArray)

		elseif (not isFirstValueATensor) and (isSecondValueATensor) then

			local dimensionSizeArray = {}

			getDimensionSizeArray(otherTensor, dimensionSizeArray)

			tensor = applyFunctionWhenTheFirstValueIsAScalar(functionToApply, tensor, otherTensor, dimensionSizeArray)

		elseif (isFirstValueATensor) and (not isSecondValueATensor) then

			local dimensionSizeArray = {}

			getDimensionSizeArray(tensor, dimensionSizeArray)

			tensor = applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, otherTensor, dimensionSizeArray)

		else

			tensor = functionToApply(tensor, otherTensor)

		end

	end

	return tensor

end

local function setValue(tensor, dimensionSizeArray, value, dimensionIndexArray)

	local dimensionIndex = dimensionIndexArray[1]

	local numberOfDimensionIndices = #dimensionIndexArray

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensionIndices > numberOfDimensions) then

		error("The number of indices exceeds the tensor's number of dimensions.")

	elseif (numberOfDimensions >= 2) then

		throwErrorIfDimensionSizeIndexIsOutOfBounds(dimensionIndex, 1, dimensionSizeArray[1])

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		local remainingIndexArray = removeFirstValueFromArray(dimensionIndexArray)

		setValue(tensor[dimensionIndex], remainingDimensionSizeArray, value, remainingIndexArray)

	elseif (numberOfDimensions == 1) then

		tensor[dimensionIndex] = value

	else

		error("An error has occurred when attempting to set the tensor value.")

	end

end

function AqwamTensorLibrary:getValue(tensor, dimensionIndexArray)

	local tensor = self.tensor

	local dimensionSizeArray = self.dimensionSizeArray

	local totalDimensionSize = getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

	local targetDimensionSize = getTotalSizeFromDimensionSizeArray(dimensionIndexArray)

	local totalSizeWithoutLastDimensionSize = totalDimensionSize / dimensionSizeArray[#dimensionSizeArray]

	if (targetDimensionSize > totalSizeWithoutLastDimensionSize) then return tensor[targetDimensionSize] end

end

local function sumFromAllDimensions(tensor)

	local result = 0

	for _, value in ipairs(tensor) do result = result + value end

	return result

end

local function sumAlongOneDimension(tensor, dimensionSizeArray, targetDimension)

	local resultTensor = {}

	local resultDimensionSizeArray = table.clone(dimensionSizeArray)

	resultDimensionSizeArray[targetDimension] = 1

	for currentDimension, size in ipairs(dimensionSizeArray) do

		if (currentDimension == targetDimension) then

		else



		end


	end

	return resultTensor, resultDimensionSizeArray

end

function AqwamTensorLibrary:sum(dimension)

	if (type(dimension) ~= "number") then error("The dimension must be a number.") end

	dimension = dimension or 0

	local tensor = self.tensor

	local dimensionSizeArray = self.dimensionSizeArray

	local numberOfDimensions = #dimensionSizeArray

	if (dimension == 0) then return sumFromAllDimensions(tensor) end

	throwErrorIfDimensionIsOutOfBounds(dimension, 1, numberOfDimensions)

	local resultTensor, resultDimensionSizeArray = sumAlongOneDimension(tensor, dimensionSizeArray, dimension)

	return self.construct(resultTensor, resultDimensionSizeArray)

end

local function tensorProduct(tensor1, tensor2)

	local dimensionArray1 = getDimensionSizeArray(tensor1)

	local dimensionArray2 = getDimensionSizeArray(tensor2)

	for i, _ in ipairs(dimensionArray1) do if (dimensionArray1[i] ~= dimensionArray2[i]) then error("Invalid dimensions.") end end

	local numberOfValues = dimensionArray1[1]

	local resultTensor = {}

	for i = 1, numberOfValues, 1 do

		if (#dimensionArray1 > 1) then

			local subproduct = tensorProduct(tensor1[i], tensor2[i])

			table.insert(resultTensor, subproduct)

		else

			table.insert(resultTensor, tensor1[i] * tensor2[i])

		end

	end

	return resultTensor
end

local function innerProduct(tensor1, tensor2)

	local dimensionArray1 = {}

	local dimensionArray2 = {}

	getDimensionSizeArray(tensor1, dimensionArray1)

	getDimensionSizeArray(tensor2, dimensionArray2)

	for i, _ in ipairs(dimensionArray1) do if (dimensionArray1[i] ~= dimensionArray2[i]) then error("Invalid dimensions.") end end

	local numberOfValues = dimensionArray1[1]

	local resultTensor = 0

	for i = 1, numberOfValues, 1 do  

		if (#dimensionArray1 > 1) then

			resultTensor += innerProduct(tensor1[i], tensor2[i])

		else

			resultTensor += (tensor1[i] * tensor2[i])

		end

	end

	return resultTensor

end

local function outerProduct(tensor1, tensor2)

	local dimensionArray1 = {}

	local dimensionArray2 = {}

	getDimensionSizeArray(tensor1, dimensionArray1)

	getDimensionSizeArray(tensor2, dimensionArray2)

	for i, _ in ipairs(dimensionArray1) do if dimensionArray1[i] ~= dimensionArray2[i] then error("Invalid dimensions.") end end

	local numberOfValues = dimensionArray1[1]

	local resultTensor = {}

	for i = 1, numberOfValues do

		if (#dimensionArray1 > 1) then

			resultTensor[i] = outerProduct(tensor1[i], tensor2[i])

		else

			resultTensor[i] = {}

			for j = 1, numberOfValues do resultTensor[i][j] = tensor1[i] * tensor2[j] end

		end

	end

	return resultTensor

end

local function eq(booleanTensor)

	local dimensionArray = {}

	getDimensionSizeArray(booleanTensor, dimensionArray)

	local numberOfValues = dimensionArray[1]

	local resultTensor = true

	if (#dimensionArray > 1) then

		for i = 1, numberOfValues do resultTensor = eq(booleanTensor[i]) end

	else

		for i = 1, numberOfValues do 

			resultTensor = (resultTensor == booleanTensor[i])

			if (resultTensor == false) then return false end

		end

	end

	return resultTensor

end

function AqwamTensorLibrary.new(tensor)

	local self = setmetatable({}, AqwamTensorLibrary)

	self.tensor = tensor

	return self

end

function AqwamTensorLibrary.construct(tensor, dimensionSizeArray)
	
	local self = setmetatable({}, AqwamTensorLibrary)
	
	self.dimensionSizeArray = dimensionSizeArray

	self.tensor = tensor

	return self

end

function AqwamTensorLibrary.createTensor(dimensionSizeArray, initialValue)

	initialValue = initialValue or 0

	local self = setmetatable({}, AqwamTensorLibrary)

	local totalDimensionSize = getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

	local tensor = table.create(totalDimensionSize, initialValue)

	self.dimensionSizeArray = dimensionSizeArray

	self.tensor = tensor

	return self

end

local function truncateDimensionSizeArrayIfRequired(dimensionSizeArray)

	local newDimensionSizeArray = table.clone(dimensionSizeArray)

	local numberOfStartingDimensionsWithTheSizeOf1 = 0

	while true do

		local size = newDimensionSizeArray[1]

		if (size ~= 1) then break end

		table.remove(newDimensionSizeArray, 1)

		numberOfStartingDimensionsWithTheSizeOf1 = numberOfStartingDimensionsWithTheSizeOf1 + 1

	end

	return newDimensionSizeArray, numberOfStartingDimensionsWithTheSizeOf1

end

function AqwamTensorLibrary.createIdentityTensor(dimensionSizeArray)

	local self = setmetatable({}, AqwamTensorLibrary)

	local totalDimensionSize = getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

	local tensor = table.create(totalDimensionSize, 0)

	for i = 1, totalDimensionSize, 1 do 

		local index = math.pow(i, i)

		if (index > totalDimensionSize) then break end

		tensor[index] = 1 

	end

	self.dimensionSizeArray = dimensionSizeArray

	self.tensor = tensor

	return self

end

function AqwamTensorLibrary.createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)

	mean = mean or 0

	standardDeviation = standardDeviation or 1

	local self = setmetatable({}, AqwamTensorLibrary)

	local totalDimensionSize = getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

	local tensor = table.create(totalDimensionSize, true)

	for i = 1, tensor, 1 do

		local randomNumber1 = math.random()

		local randomNumber2 = math.random()

		local zScore = math.sqrt(-2 * math.log(randomNumber1)) * math.cos(2 * math.pi * randomNumber2) -- Box–Muller transform formula.

		tensor[i] = (zScore * standardDeviation) + mean

	end

	self.dimensionSizeArray = dimensionSizeArray

	self.tensor = tensor

	return self

end

function AqwamTensorLibrary.createRandomUniformTensor(dimensionSizeArray, minimumValue, maximumValue)

	local self = setmetatable({}, AqwamTensorLibrary)

	local totalDimensionSize = getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

	local tensor = table.create(totalDimensionSize, true)

	for i = 1, totalDimensionSize, 1 do

		if (minimumValue) and (maximumValue) then

			tensor[i] = math.random(minimumValue, maximumValue)

		elseif (minimumValue) and (not maximumValue) then

			tensor[i] = math.random(minimumValue)

		elseif (not minimumValue) and (not maximumValue) then

			tensor[i] = math.random()

		elseif (not minimumValue) and (maximumValue) then

			error("Invalid minimum value.")

		else

			error("An unknown error has occured when creating the random uniform tensor.")

		end

	end

	self.dimensionSizeArray = dimensionSizeArray

	self.tensor = tensor

	return self

end

function AqwamTensorLibrary:getNumberOfDimensions()

	return #self.dimensionSizeArray

end

function AqwamTensorLibrary:print()

	print(self)

end

function AqwamTensorLibrary:transpose(dimensionArray)

	if (#dimensionArray ~= 2) then error("Dimension array must contain 2 dimensions.") end

	local dimension1 = dimensionArray[1]

	local dimension2 = dimensionArray[2]

	local numberOfDimensions = self:getNumberOfDimensions()

	if (dimension1 <= 0) then error("The first dimension must be greater than zero.") end

	if (dimension2 <= 0) then error("The second dimension must be greater than zero.") end

	if (dimension1 > numberOfDimensions) then error("The first dimension exceeds the tensor's number of dimensions") end

	if (dimension2 > numberOfDimensions) then error("The second dimension exceeds the tensor's number of dimensions") end

	if (dimension1 == dimension2) then error("The first dimension is equal to the second dimension.") end

	local dimensionSizeArray = self.dimensionSizeArray

	local transposedDimensionSizeArray = table.clone(dimensionSizeArray)

	local dimensionSize1 = dimensionSizeArray[dimension1]

	local dimensionSize2 = dimensionSizeArray[dimension2]

	transposedDimensionSizeArray[dimension1] = dimensionSize2

	transposedDimensionSizeArray[dimension2] = dimensionSize1

	local transposedTensor = deepCopyTable(self.tensor)

	return AqwamTensorLibrary.construct(transposedTensor, transposedDimensionSizeArray)

end

function AqwamTensorLibrary:__eq(other)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a == b) end, self, other)

	local isEqual = eq(resultTensor)

	return isEqual

end

function AqwamTensorLibrary:isEqualTo(other)

	local resultTensor, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a == b) end, self, other)

	return AqwamTensorLibrary.construct(resultTensor, dimensionSizeArray)

end

function AqwamTensorLibrary:isGreaterThan(other)

	local resultTensor, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a > b) end, self, other)

	return AqwamTensorLibrary.construct(resultTensor, dimensionSizeArray)

end

function AqwamTensorLibrary:isGreaterOrEqualTo(other)

	local resultTensor, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a >= b) end, self, other)

	return AqwamTensorLibrary.construct(resultTensor, dimensionSizeArray)

end

function AqwamTensorLibrary:isLessThan(other)

	local resultTensor, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a < b) end, self, other)

	return AqwamTensorLibrary.construct(resultTensor, dimensionSizeArray)

end

function AqwamTensorLibrary:isLessOrEqualTo(other)

	local resultTensor, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a <= b) end, self, other)

	return AqwamTensorLibrary.construct(resultTensor)

end

function AqwamTensorLibrary:tensorProduct(other)

	local resultTensor, dimensionSizeArray = tensorProduct(self, other)

	return AqwamTensorLibrary.construct(resultTensor, dimensionSizeArray)

end

function AqwamTensorLibrary:innerProduct(other)

	return innerProduct(self, other)

end

function AqwamTensorLibrary:outerProduct(other)

	local resultTensor = outerProduct(self, other)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:copy()

	return deepCopyTable(self)

end

function AqwamTensorLibrary:rawCopy()

	return deepCopyTable(self.tensor)

end

function AqwamTensorLibrary:__add(other)

	local resultTensor, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a + b) end, self, other)

	return AqwamTensorLibrary.construct(resultTensor, dimensionSizeArray)

end

function AqwamTensorLibrary:add(...)

	local resultTensor, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a + b) end, self, ...)

	return AqwamTensorLibrary.construct(resultTensor, dimensionSizeArray)

end

function AqwamTensorLibrary:__sub(other)

	local resultTensor, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a - b) end, self, other)

	return AqwamTensorLibrary.construct(resultTensor, dimensionSizeArray)

end

function AqwamTensorLibrary:subtract(...)

	local resultTensor, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a - b) end, self, ...)

	return AqwamTensorLibrary.construct(resultTensor, dimensionSizeArray)

end

function AqwamTensorLibrary:__mul(other)

	local resultTensor, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a * b) end, self, other)

	return AqwamTensorLibrary.construct(resultTensor, dimensionSizeArray)

end

function AqwamTensorLibrary:multiply(...)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a * b) end, self, ...)

	return AqwamTensorLibrary.construct(resultTensor)

end

function AqwamTensorLibrary:__div(other)

	local resultTensor, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a / b) end, self, other)

	return AqwamTensorLibrary.construct(resultTensor, dimensionSizeArray)

end

function AqwamTensorLibrary:divide(...)

	local resultTensor, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a, b) return (a / b) end, self, ...)

	return AqwamTensorLibrary.construct(resultTensor, dimensionSizeArray)

end

function AqwamTensorLibrary:__unm()

	local resultTensor, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a) return (-a) end, self)

	return AqwamTensorLibrary.construct(resultTensor, dimensionSizeArray)

end

function AqwamTensorLibrary:unaryMinus()

	local resultTensor, dimensionSizeArray = applyFunctionOnMultipleTensors(function(a) return (-a) end, self)

	return AqwamTensorLibrary.construct(resultTensor, dimensionSizeArray)

end

function AqwamTensorLibrary:__tostring()

	return self:generateTensorString()

end

function AqwamTensorLibrary:__len()

	return #self.tensor

end

function AqwamTensorLibrary:__index(index)

	if (typeof(index) == "number") then

		return rawget(self.tensor, index)

	else

		return rawget(AqwamTensorLibrary, index)

	end

end

function AqwamTensorLibrary:__newindex(index, value)

	rawset(self, index, value)

end

local function getOutOfBoundsIndexArray(array, arrayToBeCheckedForOutOfBounds)

	local outOfBoundsIndexArray = {}

	for i, value in ipairs(arrayToBeCheckedForOutOfBounds) do

		if (value < 1) or (value > array[i]) then table.insert(outOfBoundsIndexArray, i) end

	end

	return outOfBoundsIndexArray

end

function AqwamTensorLibrary:extract(originDimensionIndexArray, targetDimensionIndexArray)

	local dimensionSizeArray = self.dimensionSizeArray

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

	local tensor = self.tensor

	local newDimensionSizeArray = table.clone(targetDimensionIndexArray)

	for i, size in ipairs(newDimensionSizeArray) do newDimensionSizeArray[i] = size - originDimensionIndexArray[i] end

	local extractedTensor = {}

	local originTotalDimensionSize = getTotalSizeFromDimensionSizeArray(originDimensionIndexArray)

	local targetTotalDimensionSize = getTotalSizeFromDimensionSizeArray(targetDimensionIndexArray)

	for i = originTotalDimensionSize, targetTotalDimensionSize do table.insert(extractedTensor, tensor[i]) end

	return self.construct(extractedTensor, newDimensionSizeArray)

end

local function dotProduct(tensor1, tensor2, tensor1DimensionSizeArray, tensor2DimensionSizeArray) -- Best one. Do not delete!

	local numberOfDimensions1 = #tensor1DimensionSizeArray

	local numberOfDimensions2 = #tensor2DimensionSizeArray

	local tensor = {}

	if (numberOfDimensions1 == 1) and (numberOfDimensions2 == 2) then

		for i = 1, #tensor1, 1 do -- Last dimension, so represents columns.

			tensor[i] = 0

			for j = 1, #tensor2[1], 1 do tensor[i] = (tensor1[i] * tensor2[i][j]) end -- Since tensor 1 column size matches with tensor 2 row size, we can use column index from tensor 1.

		end

	elseif (numberOfDimensions1 == 2) and (numberOfDimensions2 == 2) then

		local tensor1Row = #tensor1

		local tensor1Column = #tensor1[1]

		local tensor2Column = #tensor2[1]

		for row = 1, tensor1Row, 1 do

			tensor[row] = {}

			for column = 1, tensor2Column, 1 do

				local sum = 0

				for i = 1, tensor1Column do sum = sum + (tensor1[row][i] * tensor2[i][column]) end

				tensor[row][column] = sum

			end

		end

	elseif (numberOfDimensions1 > 1) and (numberOfDimensions2 > 2) then

		local remainingTensor1DimensionSizeArray = removeFirstValueFromArray(tensor1DimensionSizeArray)

		local remainingTensor2DimensionSizeArray = removeFirstValueFromArray(tensor2DimensionSizeArray)

		for i = 1, tensor1DimensionSizeArray[1] do tensor[i] = dotProduct(tensor1[i], tensor2[i], remainingTensor1DimensionSizeArray, remainingTensor2DimensionSizeArray) end

	elseif (numberOfDimensions1 > 1) and (numberOfDimensions2 == 2) then

		local remainingTensor1DimensionSizeArray = removeFirstValueFromArray(tensor1DimensionSizeArray)

		for i = 1, tensor1DimensionSizeArray[1] do tensor = dotProduct(tensor1[i], tensor2, remainingTensor1DimensionSizeArray, tensor2DimensionSizeArray) end

	elseif (numberOfDimensions1 == 1) and (numberOfDimensions2 > 2) then

		local remainingTensor2DimensionSizeArray = removeFirstValueFromArray(tensor2DimensionSizeArray)

		for i = 1, tensor2DimensionSizeArray[1] do tensor = dotProduct(tensor1, tensor2[i], tensor1DimensionSizeArray, remainingTensor2DimensionSizeArray) end

	elseif (numberOfDimensions1 > 1) and (numberOfDimensions2 == 1) then

		for i = 1, tensor1DimensionSizeArray[1], 1 do

			for j = 1, tensor1DimensionSizeArray[2], 1 do 

				tensor[i] = {}

				local sum = 0

				for k = 1, tensor2DimensionSizeArray[1] do

					sum = sum + (tensor1[i][j] * tensor2[k]) 

				end

				tensor[i][j] = sum

			end

		end

	elseif (numberOfDimensions1 == 0) or (numberOfDimensions2 == 0) then

		tensor = AqwamTensorLibrary:multiply(tensor1, tensor2)

	else

		error({numberOfDimensions1, numberOfDimensions2})

	end

	return tensor

end

local function tensor2DimensionalDotProduct(tensor1, tensor2)

	local subTensor = {}

	local tensor1Row = #tensor1

	local tensor1Column = #tensor1[1]

	local tensor2Row = #tensor2

	local tensor2Column = #tensor2[1]

	if (tensor1Column ~= tensor2Row) then error("Unable to perform the dot product. The size of second last dimension of the first tensor does not equal to the size of the last dimension of the second tensor.") end

	for row = 1, tensor1Row, 1 do

		subTensor[row] = {}

		for column = 1, tensor2Column, 1 do

			local sum = 0

			for i = 1, tensor1Column do sum = sum + (tensor1[row][i] * tensor2[i][column]) end

			subTensor[row][column] = sum

		end

	end

	return subTensor

end

local function recursiveExpandedDotProduct(tensor1, tensor2, dimensionSizeArray1, dimensionSizeArray2) -- Since both have equal number of dimensions now, we only need to use only one dimension size array.

	local numberOfDimensions1 = #dimensionSizeArray1

	local numberOfDimensions2 = #dimensionSizeArray2

	local tensor

	if (numberOfDimensions1 >= 3) and (numberOfDimensions2 >= 3) and (dimensionSizeArray1[1] == dimensionSizeArray2[1]) then

		tensor = {}

		local remainingDimensionSizeArray1 = removeFirstValueFromArray(dimensionSizeArray1)

		local remainingDimensionSizeArray2 = removeFirstValueFromArray(dimensionSizeArray2)

		for i = 1, dimensionSizeArray1[1], 1 do tensor[i] = recursiveExpandedDotProduct(tensor1[i], tensor2[i], remainingDimensionSizeArray1, remainingDimensionSizeArray2) end

	elseif (numberOfDimensions1 == 2) and (numberOfDimensions2 == 2) and (dimensionSizeArray1[2] == dimensionSizeArray2[1]) then -- No need an elseif statement where number of dimension is 1. This operation requires 2D tensors.

		tensor = tensor2DimensionalDotProduct(tensor1, tensor2)

	elseif (numberOfDimensions1 == 0) or (numberOfDimensions2 == 0) then

		tensor = AqwamTensorLibrary:multiply(tensor1, tensor2)

	elseif (numberOfDimensions1 >= 2) and (numberOfDimensions2 >= 2) and (dimensionSizeArray1[1] ~= dimensionSizeArray2[1]) then

		error("Unable to dot product. The starting dimension sizes of the first tensor does not equal to the starting dimension sizes of the second tensor.")

	else

		error("Unable to dot product.")

	end

	return tensor

end

local function expandedDotProduct(tensor1, tensor2)

	local dimensionSizeArray1 =  AqwamTensorLibrary:getDimensionSizeArray(tensor1)

	local dimensionSizeArray2 =  AqwamTensorLibrary:getDimensionSizeArray(tensor2)

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

		expandedTensor1 = AqwamTensorLibrary:increaseNumberOfDimensions(tensor1, dimensionSizeToAddArray)

	else

		expandedTensor1 = tensor1

	end

	if (numberOfDimensionsOffset2 ~= 0) then

		local dimensionSizeToAddArray = {}

		for i = 1, numberOfDimensionsOffset2, 1 do table.insert(dimensionSizeToAddArray, dimensionSizeArray1[i]) end

		expandedTensor2 = AqwamTensorLibrary:increaseNumberOfDimensions(tensor2, dimensionSizeToAddArray)

	else

		expandedTensor2 = tensor2

	end

	local expandedTensor1DimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(expandedTensor1)

	local expandedTensor2DimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(expandedTensor2)

	return recursiveExpandedDotProduct(expandedTensor1, expandedTensor2, expandedTensor1DimensionSizeArray, expandedTensor2DimensionSizeArray)

end

local function hardcodedDotProduct(tensor1, tensor2)

	local numberOfDimensions1 = AqwamTensorLibrary:getNumberOfDimensions(tensor1)

	local numberOfDimensions2 = AqwamTensorLibrary:getNumberOfDimensions(tensor2)

	local numberOfDimensionsOffset1 = 5 - numberOfDimensions1

	local numberOfDimensionsOffset2 = 5 - numberOfDimensions2

	local expandedTensor1 = AqwamTensorLibrary:increaseNumberOfDimensions(tensor1, table.create(numberOfDimensionsOffset1, 1))

	local expandedTensor2 = AqwamTensorLibrary:increaseNumberOfDimensions(tensor2, table.create(numberOfDimensionsOffset2, 1))

	local expandedNumberOfDimension1 = AqwamTensorLibrary:getDimensionSizeArray(expandedTensor1)

	local expandedNumberOfDimension2 = AqwamTensorLibrary:getDimensionSizeArray(expandedTensor2)

	local tensor = {}

	for a = 1, expandedNumberOfDimension1[1], 1 do

		tensor[a] = {}

		for b = 1, expandedNumberOfDimension1[2], 1 do

			tensor[a][b] = {}

			for c = 1, expandedNumberOfDimension1[3], 1 do

				tensor[a][b][c] = {}

				for d = 1, expandedNumberOfDimension1[4], 1 do

					tensor[a][b][c][d] = {}

					for e = 1, expandedNumberOfDimension2[5], 1 do

						tensor[a][b][c][d][e] = {}

						local sum = 0

						for f = 1, expandedNumberOfDimension1[5] do sum = sum + (expandedTensor1[a][b][c][d][f] * expandedTensor2[a][b][c][f][e]) end

						tensor[a][b][c][d][e] = sum

					end

				end

			end

		end

	end

	return tensor

end

function AqwamTensorLibrary:dotProduct(other) -- Refer to this article. It was a fucking headache to do this. https://medium.com/@hunter-j-phillips/a-simple-introduction-to-tensors-c4a8321efffc

	local resultTensor = expandedDotProduct(self, other)

	return AqwamTensorLibrary.new(resultTensor)

end

local function get2DTensorTextSpacing(tensor, dimensionSizeArray, textSpacingArray) -- Dimension size array is put here because it is computationally expensive to use recurvsive just to get the dimension size.

	if (#dimensionSizeArray > 1) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do textSpacingArray = get2DTensorTextSpacing(tensor[i], remainingDimensionSizeArray, textSpacingArray) end

	else

		for i = 1, dimensionSizeArray[1], 1 do textSpacingArray[i] = math.max(textSpacingArray[i], string.len(tostring(tensor[i]))) end

	end

	return textSpacingArray

end

function AqwamTensorLibrary:get2DTensorTextSpacing()

	local dimensionSizeArray = self:getDimensionSizeArray()

	local numberOfDimensions = #dimensionSizeArray

	local sizeAtFinalDimension = dimensionSizeArray[numberOfDimensions]

	local textSpacingArray = table.create(sizeAtFinalDimension, 0)

	return get2DTensorTextSpacing(self, dimensionSizeArray, textSpacingArray)

end

local function generateTensorString(tensor, dimensionSizeArray, textSpacingArray, dimensionDepth)

	local numberOfDimensions = #dimensionSizeArray

	local tensorLength = #tensor

	local text = " "

	if (numberOfDimensions > 1) then

		local spacing = ""

		text = text .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end

		local remainingDimensionSizeArray = removeLastValueFromArray(dimensionSizeArray)

		for i = 1, #tensor do

			if (i > 1) then text = text .. spacing end

			text = text .. generateTensorString(tensor[i], remainingDimensionSizeArray, textSpacingArray, dimensionDepth + 1)

			if (i == tensorLength) then continue end

			text = text .. "\n"

		end

		text = text .. " }"

	else

		text = text .. "{ "

		for i = 1, tensorLength do

			local cellValue = tensor[i]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = textSpacingArray[i] - cellWidth

			text = text .. string.rep(" ", padding) .. cellText

			if (i == tensorLength) then continue end

			text = text .. " "

		end

		text = text .. " }"

	end

	return text

end

function AqwamTensorLibrary:generateTensorString()

	local dimensionSizeArray = self:getDimensionSizeArray()

	local textSpacingArray = self:get2DTensorTextSpacing()

	return generateTensorString(self, dimensionSizeArray, textSpacingArray, 1)

end

local function generateTensorStringWithComma(tensor, dimensionSizeArray, textSpacingArray, dimensionDepth)

	local numberOfDimensions = #dimensionSizeArray

	local tensorLength = #tensor

	local text = " "

	if (numberOfDimensions > 1) then

		local spacing = ""

		text = text .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end

		local remainingDimensionSizeArray = removeLastValueFromArray(dimensionSizeArray)

		for i = 1, #tensor do

			if (i > 1) then text = text .. spacing end

			text = text .. generateTensorStringWithComma(tensor[i], remainingDimensionSizeArray, textSpacingArray, dimensionDepth + 1)

			if (i == tensorLength) then continue end

			text = text .. "\n"

		end

		text = text .. " }"

	else

		text = text .. "{ "

		for i = 1, tensorLength do 

			local cellValue = tensor[i]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = textSpacingArray[i] - cellWidth

			text = text .. string.rep(" ", padding) .. cellText

			if (i == tensorLength) then continue end

			text = text .. ", "

		end

		text = text .. " }"

	end

	return text

end

function AqwamTensorLibrary:generateTensorStringWithComma()

	local dimensionSizeArray = self:getDimensionSizeArray()

	local textSpacingArray = self:get2DTensorTextSpacing()

	return generateTensorStringWithComma(self, dimensionSizeArray, textSpacingArray, 1)

end

local function generatePortableTensorString(tensor, dimensionSizeArray, textSpacingArray, dimensionDepth)

	local numberOfDimensions = #dimensionSizeArray

	local tensorLength = #tensor

	local text = " "

	if (numberOfDimensions > 1) then

		local spacing = ""

		text = text .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end

		local remainingDimensionSizeArray = removeLastValueFromArray(dimensionSizeArray)

		for i = 1, #tensor do

			if (i > 1) then text = text .. spacing end

			text = text .. generatePortableTensorString(tensor[i], remainingDimensionSizeArray, textSpacingArray, dimensionDepth + 1)

			if (i == tensorLength) then continue end

			text = text .. "\n"

		end

		text = text .. " }"

		if (dimensionDepth > 1) then text = text .. "," end

	else

		text = text .. "{ "

		for i = 1, tensorLength do 

			local cellValue = tensor[i]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = textSpacingArray[i] - cellWidth

			text = text .. string.rep(" ", padding) .. cellText

			if (i == tensorLength) then continue end

			text = text .. ", "

		end

		text = text .. " },"

	end

	return text

end

function AqwamTensorLibrary:generatePortableTensorString()

	local dimensionSizeArray = self:getDimensionSizeArray()

	local textSpacingArray = self:get2DTensorTextSpacing()

	return generatePortableTensorString(self, dimensionSizeArray, textSpacingArray, 1)

end

function AqwamTensorLibrary:printTensor()

	print("\n\n" .. self:generateTensorString() .. "\n\n")

end

function AqwamTensorLibrary:printTensorWithComma()

	print("\n\n" .. self:generateTensorStringWithComma() .. "\n\n")

end

function AqwamTensorLibrary:printPortableTensor()

	print("\n\n" .. self:generatePortableTensorString() .. "\n\n")

end

local function flattenIntoASingleDimension(tensor, dimensionSizeArray, targetTensor)

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do flattenIntoASingleDimension(tensor[i], remainingDimensionSizeArray, targetTensor) end

	else

		for _, value in ipairs(tensor) do table.insert(targetTensor, value) end

	end

	return tensor

end

local function flattenAlongSpecifiedDimensions(tensor, dimensionSizeArray, startDimension, endDimension)

	local numberOfDimensions = #dimensionSizeArray

	local flattenedDimensionSize = 1

	local newDimensionSizeArray = {}

	for currentDimension = numberOfDimensions, 1, -1 do

		local currentDimensionSize = dimensionSizeArray[currentDimension]

		if (currentDimension > startDimension) and (currentDimension <= endDimension) then 

			flattenedDimensionSize = flattenedDimensionSize * currentDimensionSize

		elseif (currentDimension == startDimension) then

			flattenedDimensionSize = flattenedDimensionSize * currentDimensionSize

			table.insert(newDimensionSizeArray, 1, flattenedDimensionSize)

		else

			table.insert(newDimensionSizeArray, 1, currentDimensionSize)

		end

	end

	return tensor:reshape(newDimensionSizeArray)

end

function AqwamTensorLibrary:flatten(startDimension, endDimension)

	local dimensionSizeArray = self:getDimensionSizeArray()

	local flattenedTensor

	if (not startDimension) and (not endDimension) then

		flattenedTensor = {}

		flattenIntoASingleDimension(self, dimensionSizeArray, flattenedTensor)

	else

		startDimension = startDimension or 1

		endDimension = endDimension or math.huge

		flattenedTensor = flattenAlongSpecifiedDimensions(self, dimensionSizeArray, startDimension, endDimension)

	end

	return AqwamTensorLibrary.new(flattenedTensor)

end

local function reshapeFromFlattenedTensor(tensor, dimensionSizeArray, dimensionIndex)

	local resultTensor = {}

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do 

			resultTensor[i], dimensionIndex = reshapeFromFlattenedTensor(tensor, remainingDimensionSizeArray, dimensionIndex) 

		end

	else

		for i = 1, dimensionSizeArray[1], 1 do 

			table.insert(resultTensor, tensor[dimensionIndex])
			dimensionIndex = dimensionIndex + 1

		end

	end

	return resultTensor, dimensionIndex

end

local function incrementFinalDimensionIndex(targetDimensionSizeArray, currentTargetDimensionIndexArray)

	local numberOfDimensions = #currentTargetDimensionIndexArray

	currentTargetDimensionIndexArray[numberOfDimensions] = currentTargetDimensionIndexArray[numberOfDimensions] + 1

	for dimension = numberOfDimensions, 1, -1 do

		if ((targetDimensionSizeArray[dimension] + 1) == currentTargetDimensionIndexArray[dimension]) then

			currentTargetDimensionIndexArray[dimension] = 1

			if (dimension >= 2) then currentTargetDimensionIndexArray[dimension - 1] = currentTargetDimensionIndexArray[dimension - 1] + 1 end

		end	

	end

	return currentTargetDimensionIndexArray

end

local function reshape(tensor, dimensionSizeArray, targetTensor, targetDimensionSizeArray, currentTargetDimensionIndexArray)

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do 

			currentTargetDimensionIndexArray = reshape(tensor[i], remainingDimensionSizeArray, targetTensor, targetDimensionSizeArray, currentTargetDimensionIndexArray) 

		end

	else

		for i = 1, dimensionSizeArray[1], 1 do 

			targetTensor:setValue(tensor[i], currentTargetDimensionIndexArray)

			currentTargetDimensionIndexArray = incrementFinalDimensionIndex(targetDimensionSizeArray, currentTargetDimensionIndexArray)

		end

	end

	return currentTargetDimensionIndexArray

end

function AqwamTensorLibrary:inefficientReshape(dimensionSizeArray) -- This one requires higher space complexity due to storing the target dimension index array for each of the values. It is also less efficient because it needs to use recursion to get and set values from and to the target tensor.

	local tensorDimensionSizeArray = self:getDimensionSizeArray()

	local totalNumberOfValue = getTotalSizeFromDimensionSizeArray(tensorDimensionSizeArray)

	local totalNumberOfValuesRequired = getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

	if (totalNumberOfValue ~= totalNumberOfValuesRequired) then error("The number of values of the tensor does not equal to total number of values of the reshaped tensor.") end

	local numberOfDimensions = #tensorDimensionSizeArray

	local resultTensor

	if (numberOfDimensions == 1) then

		resultTensor = reshapeFromFlattenedTensor(self, dimensionSizeArray, 1)

	else

		resultTensor = AqwamTensorLibrary:createTensor(dimensionSizeArray, true)

		local currentTargetDimensionIndexArray = table.create(#dimensionSizeArray, 1)

		reshape(self, tensorDimensionSizeArray, resultTensor, dimensionSizeArray, currentTargetDimensionIndexArray)

	end

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:reshape(dimensionSizeArray) -- This one requires lower space complexity as it only need to flatten the tensor. Then only need a single target dimension index array that will be used by all values from the original tebsor.

	local tensorDimensionSizeArray = self:getDimensionSizeArray()

	local totalNumberOfValue = getTotalSizeFromDimensionSizeArray(tensorDimensionSizeArray)

	local totalNumberOfValuesRequired = getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

	if (totalNumberOfValue ~= totalNumberOfValuesRequired) then error("The number of values of the tensor does not equal to total number of values of the reshaped tensor.") end

	local numberOfDimensions = #tensorDimensionSizeArray

	local resultTensor

	if (numberOfDimensions >= 2) then resultTensor = self:flatten() end

	resultTensor = reshapeFromFlattenedTensor(resultTensor, dimensionSizeArray, 1)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:truncate(numberOfDimensionsToTruncate)

	numberOfDimensionsToTruncate = numberOfDimensionsToTruncate or math.huge

	if (numberOfDimensionsToTruncate ~= math.huge) and (numberOfDimensionsToTruncate ~= nil) then

		local dimensionSizeArray = self:getDimensionSizeArray()

		for dimension = 1, numberOfDimensionsToTruncate, 1 do

			local size = dimensionSizeArray[dimension]

			if (size ~= 1) then error("Unable to truncate. Dimension " .. dimension .. " has the size of " .. size .. ".") end

		end

	end

	local resultTensor = deepCopyTable(self.tensor)

	for dimension = 1, numberOfDimensionsToTruncate, 1 do

		if (type(resultTensor) ~= "table") then break end

		if (#resultTensor ~= 1) then break end

		resultTensor = resultTensor[1]

	end

	return AqwamTensorLibrary.new(resultTensor)

end

local function squeeze(tensor, dimensionSizeArray, targetDimension, currentDimension)

	local numberOfDimensions = #dimensionSizeArray

	local isAtTargetDimension = (currentDimension == targetDimension)

	local isATensor = (numberOfDimensions >= 1)

	local remainingDimensionSizeArray

	local resultTensor

	if (isAtTargetDimension) and (isATensor) then

		resultTensor = {}

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		remainingDimensionSizeArray = removeFirstValueFromArray(remainingDimensionSizeArray)

		for i = 1, dimensionSizeArray[2], 1 do resultTensor[i] = squeeze(tensor[1][i], remainingDimensionSizeArray, targetDimension, currentDimension + 2) end

	elseif (not isAtTargetDimension) and (isATensor) then

		resultTensor = {}

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do resultTensor[i] = squeeze(tensor[i], remainingDimensionSizeArray, targetDimension, currentDimension + 1) end

	elseif (not isATensor) then

		resultTensor = tensor

	else

		error("Unable to squeeze.")

	end

	return resultTensor

end

function AqwamTensorLibrary:squeeze(dimension)

	if (type(dimension) ~= "number") then error("The dimension must be a number.") end

	local dimensionSizeArray = table.clone(self.dimensionSizeArray)

	if (dimensionSizeArray[dimension] ~= 1) then error("The dimension size at dimension " .. dimension .. " is not equal to 1.") end

	table.remove(dimensionSizeArray, dimension)

	local tensor = deepCopyTable(self.tensor)

	return self.construct(tensor, dimensionSizeArray)

end

function AqwamTensorLibrary:destroy()

	self.tensor = nil

	setmetatable(self, nil)

end

return AqwamTensorLibrary
