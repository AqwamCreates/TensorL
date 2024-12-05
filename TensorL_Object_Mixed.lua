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

local AqwamTensorLibrary = {}

local function incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	for i = #dimensionIndexArray, 1, -1 do

		dimensionIndexArray[i] = dimensionIndexArray[i] + 1

		if (dimensionIndexArray[i] <= dimensionSizeArray[i]) then break end

		dimensionIndexArray[i] = 1

	end

	return dimensionIndexArray

end

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

local function throwErrorIfDimensionIndexIsOutOfBounds(dimensionSizeIndex, minimumDimensionSizeIndex, maximumDimensionSizeIndex)

	if checkIfValueIsOutOfBounds(dimensionSizeIndex, minimumDimensionSizeIndex, maximumDimensionSizeIndex) then error("The dimension index is out of bounds.") end

end

local function throwErrorIfDimensionIsOutOfBounds(dimension, minimumNumberOfDimensions, maximumNumberOfDimensions)

	if checkIfValueIsOutOfBounds(dimension, minimumNumberOfDimensions, maximumNumberOfDimensions) then error("The dimension is out of bounds.") end

end

local function removeFirstValueFromArray(array)

	local newArray = {}

	for i = 2, #array, 1 do table.insert(newArray, array[i]) end

	return newArray

end

local function createTensor(dimensionSizeArray, initialValue) -- Don't put dimension size array truncation here. It is needed for several operations like dot product. 

	local tensor = {}

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = createTensor(remainingDimensionSizeArray, initialValue) end

	else

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = initialValue end

	end

	return tensor

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

	if (type(tensor) ~= "table") then return 0 end

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

	getDimensionSizeArrayRecursive(self, dimensionSizeArray)

	return dimensionSizeArray

end

local function expandDimensionSizes(tensor, dimensionSizeArray, targetDimensionSizeArray)

	-- Does not do the same thing with inefficient expandDimensionSizes function. This one expandDimensionSizes at the lowest dimension first and then the parent dimension will make copy of this.

	local resultTensor

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensions >= 2) then

		resultTensor = {}

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		local remainingTargetDimensionSizeArray = removeFirstValueFromArray(targetDimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do resultTensor[i] = expandDimensionSizes(tensor[i], remainingDimensionSizeArray, remainingTargetDimensionSizeArray) end

	else

		resultTensor = deepCopyTable(tensor)  -- If the "(numberOfDimensions > 1)" from the first "if" statement does not run, it will return the original tensor. So we need to deep copy it.

	end

	local dimensionSize = #resultTensor -- Need to call this again because we may have modified the tensor below it that leads to the change of the dimension size array.

	local targetDimensionSize = targetDimensionSizeArray[1]

	local hasSameDimensionSize = (dimensionSize == targetDimensionSize)

	local canDimensionBeExpanded = (dimensionSize == 1)

	if (numberOfDimensions >= 1) and (not hasSameDimensionSize) and (canDimensionBeExpanded) then 

		local subTensor = resultTensor[1]

		for i = 1, targetDimensionSize, 1 do resultTensor[i] = deepCopyTable(subTensor) end

	elseif (not hasSameDimensionSize) and (not canDimensionBeExpanded) then

		error("Unable to expand at dimension " .. numberOfDimensions .. ".")

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

local function setValue(tensor, dimensionSizeArray, value, dimensionIndexArray)

	local dimensionIndex = dimensionIndexArray[1]

	local numberOfDimensionIndices = #dimensionIndexArray

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensionIndices > numberOfDimensions) then

		error("The number of indices exceeds the tensor's number of dimensions.")

	elseif (numberOfDimensions >= 2) then

		throwErrorIfDimensionIndexIsOutOfBounds(dimensionIndex, 1, dimensionSizeArray[1])

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		local remainingIndexArray = removeFirstValueFromArray(dimensionIndexArray)

		setValue(tensor[dimensionIndex], remainingDimensionSizeArray, value, remainingIndexArray)

	elseif (numberOfDimensions == 1) then

		tensor[dimensionIndex] = value

	else

		error("An error has occurred when attempting to set the tensor value.")

	end

end

local function getValue(tensor, dimensionSizeArray, dimensionIndexArray)

	local dimensionIndex = dimensionIndexArray[1]

	local numberOfDimensionIndices = #dimensionIndexArray

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensionIndices > numberOfDimensions) then

		error("The number of indices exceeds the tensor's number of dimensions.")

	elseif (numberOfDimensions >= 2) then

		throwErrorIfDimensionIndexIsOutOfBounds(dimensionIndex, 1, dimensionSizeArray[1])

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		local remainingDimensionIndexArray = removeFirstValueFromArray(dimensionIndexArray)

		return getValue(tensor[dimensionIndex], remainingDimensionSizeArray, remainingDimensionIndexArray)

	elseif (numberOfDimensions == 1) then

		return tensor[dimensionIndex]

	else

		error("An error has occurred when attempting to get the tensor value.")

	end

end

function AqwamTensorLibrary:expandDimensionSizes(targetDimensionSizeArray)
	
	local dimensionSizeArray = self:getDimensionSizeArray()

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensions ~= #targetDimensionSizeArray) then error("The number of dimensions does not match.") end

	for i, size in ipairs(dimensionSizeArray) do

		if (size ~= targetDimensionSizeArray[i]) and (size ~= 1) then error("Unable to expand at dimension " .. i .. ".") end

	end
	
	local tensor = self.tensor

	local resultTensor

	local newSubTargetDimensionSizeArray

	local oldSubTargetDimensionSizeArray = table.clone(dimensionSizeArray)

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	for dimension = #dimensionSizeArray, 1, -1 do

		local newDimensionSize = targetDimensionSizeArray[dimension]

		local oldDimensionSize = oldSubTargetDimensionSizeArray[dimension]

		newSubTargetDimensionSizeArray = table.clone(oldSubTargetDimensionSizeArray)

		newSubTargetDimensionSizeArray[dimension] = newDimensionSize

		resultTensor = createTensor(newSubTargetDimensionSizeArray, true)

		repeat

			local subDimensionIndexArray = table.clone(dimensionIndexArray)

			if (oldDimensionSize == 1) and (newDimensionSize >= 2) then

				local value = getValue(tensor, dimensionSizeArray, subDimensionIndexArray)

				for newDimensionIndex = 1, newDimensionSize, 1 do

					subDimensionIndexArray[dimension] = newDimensionIndex

					setValue(resultTensor, dimensionSizeArray, value, subDimensionIndexArray)

				end

			else

				for newDimensionIndex = 1, newDimensionSize, 1 do

					subDimensionIndexArray[dimension] = newDimensionIndex

					local value = getValue(tensor, dimensionSizeArray, subDimensionIndexArray)

					setValue(resultTensor, dimensionSizeArray, value, subDimensionIndexArray)

				end

			end

			dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, oldSubTargetDimensionSizeArray)

		until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

		oldSubTargetDimensionSizeArray = newSubTargetDimensionSizeArray

		tensor = resultTensor

	end

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:expandNumberOfDimensions(dimensionSizeToAddArray)

	local dimensionSizeArray = self:getDimensionSizeArray()

	local resultDimensionSizeArray = {}

	for i, dimensionSize in ipairs(dimensionSizeToAddArray) do table.insert(resultDimensionSizeArray, dimensionSize) end

	for i, dimensionSize in ipairs(dimensionSizeArray) do table.insert(resultDimensionSizeArray, dimensionSize) end

	local resultNumberOfDimensions = #resultDimensionSizeArray

	local resultDimensionIndexArray = table.create(resultNumberOfDimensions, 1)

	local resultDimensionIndexArrayToEndLoop = table.create(resultNumberOfDimensions, 1)

	local dimensionIndexArray = table.create(#dimensionSizeArray, 1)

	local resultTensor = AqwamTensorLibrary:createTensor(resultDimensionSizeArray)

	repeat

		local value = self:getValue(dimensionIndexArray)

		setValue(resultTensor, resultDimensionSizeArray, value, resultDimensionIndexArray)

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

		resultDimensionIndexArray = incrementDimensionIndexArray(resultDimensionIndexArray, resultDimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(resultDimensionIndexArray, resultDimensionIndexArrayToEndLoop)

	return AqwamTensorLibrary.new(resultTensor)

end

local function broadcast(tensor1, tensor2, deepCopyOriginalTensor)

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

function AqwamTensorLibrary:broadcast(tensor1, tensor2)

	return broadcast(tensor1, tensor2, true)

end

local function applyFunctionUsingOneTensor(functionToApply, tensor)

	local dimensionSizeArray = tensor:getDimensionSizeArray(tensor)

	local resultTensor = createTensor(dimensionSizeArray, true)

	local numberOfDimensions = #dimensionSizeArray

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	repeat

		local value = tensor:getValue(dimensionIndexArray)

		local resultValue = functionToApply(value)

		setValue(resultTensor, dimensionSizeArray, resultValue, dimensionIndexArray)

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return resultTensor

end

local function applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2)

	local dimensionSizeArray = tensor1:getDimensionSizeArray()

	local resultTensor = createTensor(dimensionSizeArray, true)

	local numberOfDimensions = #dimensionSizeArray

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	repeat

		local value1 = tensor1:getValue(dimensionIndexArray)

		local value2 = tensor2:getValue(dimensionIndexArray)

		local resultValue = functionToApply(value1, value2)

		setValue(resultTensor, dimensionSizeArray, resultValue, dimensionIndexArray)

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return resultTensor

end

local function applyFunctionWhenTheFirstValueIsAScalar(functionToApply, scalar, tensor)

	local dimensionSizeArray = tensor:getDimensionSizeArray()

	local resultTensor = createTensor(dimensionSizeArray, true)

	local numberOfDimensions = #dimensionSizeArray

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	repeat

		local value = tensor:getValue(dimensionIndexArray)

		local resultValue = functionToApply(scalar, value)

		setValue(resultTensor, dimensionSizeArray, resultValue, dimensionIndexArray)

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return resultTensor

end

local function applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, scalar)

	local dimensionSizeArray = tensor:getDimensionSizeArray()

	local resultTensor = createTensor(dimensionSizeArray, true)

	local numberOfDimensions = #dimensionSizeArray

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	repeat

		local value = tensor:getValue(dimensionIndexArray)

		local resultValue = functionToApply(value, scalar)

		setValue(resultTensor, dimensionSizeArray, resultValue, dimensionIndexArray)

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return resultTensor

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

local function sumFromAllDimensions(tensor)

	local dimensionSizeArray = tensor:getDimensionSizeArray()

	local numberOfDimensions = #dimensionSizeArray

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	local result = 0

	repeat

		local value = tensor:getValue(dimensionIndexArray)

		result = result + value

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return result

end

local function sumAlongOneDimension(tensor, targetDimension)

	local dimensionSizeArray = tensor:getDimensionSizeArray()

	local numberOfDimensions = #dimensionSizeArray

	throwErrorIfDimensionIsOutOfBounds(targetDimension, 1, numberOfDimensions)

	local resultDimensionSizeArray = table.clone(dimensionSizeArray)

	resultDimensionSizeArray[targetDimension] = 1

	local resultTensor = createTensor(resultDimensionSizeArray, 0)

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	repeat

		local resultDimensionIndexArray = table.clone(dimensionIndexArray)

		resultDimensionIndexArray[targetDimension] = 1

		local result = tensor:getValue(resultDimensionIndexArray)

		local value = getValue(tensor, dimensionSizeArray, dimensionIndexArray)

		result = result + value

		setValue(tensor, resultDimensionSizeArray, result, resultDimensionIndexArray)

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return resultTensor

end

function AqwamTensorLibrary:sum(dimension)

	local dimensionSizeArray = self:getDimensionSizeArray()

	if (not dimension) then return sumFromAllDimensions(self, dimensionSizeArray) end

	if (type(dimension) ~= "number") then error("The dimension must be a number.") end

	local numberOfDimensions = #dimensionSizeArray

	throwErrorIfDimensionIsOutOfBounds(dimension, 1, numberOfDimensions)

	local sumTensor = sumAlongOneDimension(self, dimensionSizeArray, dimension, 1)

	return AqwamTensorLibrary.new(sumTensor)

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

function AqwamTensorLibrary.new(tensor)

	local self = setmetatable({}, AqwamTensorLibrary)

	self.tensor = tensor

	return self

end

function AqwamTensorLibrary.createTensor(dimensionSizeArray, initialValue)

	initialValue = initialValue or 0

	local self = setmetatable({}, AqwamTensorLibrary)

	self.tensor = createTensor(dimensionSizeArray, initialValue)

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

local function createIdentityTensor(dimensionSizeArray, dimensionIndexArray)

	local tensor = {}

	if (#dimensionSizeArray >= 2) then

		for i = 1, dimensionSizeArray[1], 1 do 

			local copiedDimensionIndexArray = table.clone(dimensionIndexArray)

			table.insert(copiedDimensionIndexArray, i)

			local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

			tensor[i] = createIdentityTensor(remainingDimensionSizeArray, copiedDimensionIndexArray) 

		end

	else

		for i = 1, dimensionSizeArray[1], 1 do

			local copiedDimensionIndexArray = table.clone(dimensionIndexArray)

			local firstDimensionIndex = copiedDimensionIndexArray[1]

			table.insert(copiedDimensionIndexArray, i)

			tensor[i] = 1

			for _, dimensionIndex in ipairs(copiedDimensionIndexArray) do

				if (dimensionIndex ~= firstDimensionIndex) then

					tensor[i] = 0
					break

				end

			end

		end

	end

	return tensor

end

function AqwamTensorLibrary.createIdentityTensor(dimensionSizeArray)

	local truncatedDimensionSizeArray, numberOfDimensionsOfSize1 = truncateDimensionSizeArrayIfRequired(dimensionSizeArray)

	--local resultTensor = createIdentityTensor(truncatedDimensionSizeArray, {})

	local truncatedNumberOfDimensions = #truncatedDimensionSizeArray

	local resultTensor = createTensor(truncatedDimensionSizeArray, 0)

	for i = 1, truncatedNumberOfDimensions, 1 do

		local canSetValueToOne = true

		for _, dimensionSize in ipairs(truncatedDimensionSizeArray) do

			if (dimensionSize < i) then

				canSetValueToOne = false
				break

			end

		end

		if (canSetValueToOne) then

			local dimensionIndexArray = table.create(truncatedNumberOfDimensions, i)

			AqwamTensorLibrary:setValue(resultTensor, 1, dimensionIndexArray)

		end

	end

	for i = 1, numberOfDimensionsOfSize1, 1 do resultTensor = {resultTensor} end

	local self = setmetatable({}, AqwamTensorLibrary)

	self.tensor = resultTensor

	return self

end

local function createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)

	local tensor = {}

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = createRandomNormalTensor(remainingDimensionSizeArray, mean, standardDeviation) end

	else

		for i = 1, dimensionSizeArray[1], 1 do 

			local randomNumber1 = math.random()

			local randomNumber2 = math.random()

			local zScore = math.sqrt(-2 * math.log(randomNumber1)) * math.cos(2 * math.pi * randomNumber2) -- Box–Muller transform formula.

			tensor[i] = (zScore * standardDeviation) + mean

		end

	end

	return tensor

end

function AqwamTensorLibrary.createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)

	mean = mean or 0

	standardDeviation = standardDeviation or 1

	local self = setmetatable({}, AqwamTensorLibrary)

	self.tensor = createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)

	return self

end

local function createRandomUniformTensor(dimensionSizeArray, minimumValue, maximumValue)

	local numberOfDimensions = #dimensionSizeArray

	local tensor = {}

	if (numberOfDimensions >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = createRandomUniformTensor(remainingDimensionSizeArray, minimumValue, maximumValue) end

	elseif (numberOfDimensions == 1) and (minimumValue) and (maximumValue) then

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = math.random(minimumValue, maximumValue) end

	elseif (numberOfDimensions == 1) and (minimumValue) and (not maximumValue) then

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = math.random(minimumValue) end

	elseif (numberOfDimensions == 1) and (not minimumValue) and (not maximumValue) then

		for i = 1, dimensionSizeArray[1], 1 do tensor[i] = math.random() end

	elseif (numberOfDimensions == 1) and (not minimumValue) and (maximumValue) then

		error("Invalid minimum value.")

	else

		error("An unknown error has occured when creating the random uniform tensor.")

	end

	return tensor

end

function AqwamTensorLibrary.createRandomUniformTensor(dimensionSizeArray, minimumValue, maximumValue)

	local self = setmetatable({}, AqwamTensorLibrary)

	self.tensor = createRandomUniformTensor(dimensionSizeArray, minimumValue, maximumValue)

	return self

end

function AqwamTensorLibrary:getNumberOfDimensions()

	return getNumberOfDimensions(self)

end

function AqwamTensorLibrary:print()

	print(self)

end

local function truncate(tensor, numberOfDimensionsToTruncate)

	numberOfDimensionsToTruncate = numberOfDimensionsToTruncate or math.huge

	if (numberOfDimensionsToTruncate ~= math.huge) and (numberOfDimensionsToTruncate ~= nil) then

		local dimensionSizeArray = getDimensionSizeArray(tensor)

		for dimension = 1, numberOfDimensionsToTruncate, 1 do

			local size = dimensionSizeArray[dimension]

			if (size ~= 1) then error("Unable to truncate. Dimension " .. dimension .. " has the size of " .. size .. ".") end

		end

	end

	local resultTensor = deepCopyTable(tensor.tensor)

	for dimension = 1, numberOfDimensionsToTruncate, 1 do

		if (type(resultTensor) ~= "table") then break end

		if (#resultTensor ~= 1) then break end

		resultTensor = resultTensor[1]

	end

	return resultTensor

end

function AqwamTensorLibrary:truncate(numberOfDimensionsToTruncate)

	local resultTensor = truncate(self, numberOfDimensionsToTruncate)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:hardcodedTranspose(dimensionArray) -- I don't think it is worth the effort to generalize to the rest of dimensions... That being said, to process videos, you need at most 5 dimensions. Don't get confused about the channels! Only number of channels are changed and not the number of dimensions of the tensor!

	if (#dimensionArray ~= 2) then error("Dimension array must contain 2 dimensions.") end

	local dimension1 = dimensionArray[1]

	local dimension2 = dimensionArray[2]

	local numberOfDimensions = self:getNumberOfDimensions()

	if (dimension1 <= 0) then error("The first dimension must be greater than zero.") end

	if (dimension2 <= 0) then error("The second dimension must be greater than zero.") end

	if (dimension1 > numberOfDimensions) then error("The first dimension exceeds the tensor's number of dimensions") end

	if (dimension2 > numberOfDimensions) then error("The second dimension exceeds the tensor's number of dimensions") end

	if (dimension1 >= 6) then error("When using the hardcoded transpose, the first dimension must be less than six.") end

	if (dimension2 >= 6) then error("When using the hardcoded transpose, the second dimension must be less than six.") end

	if (dimension1 == dimension2) then error("The first dimension is equal to the second dimension.") end

	local dimensionArray = self:getDimensionSizeArray()

	local numberOfDimensions = #dimensionArray

	local offset = 5 - numberOfDimensions

	local dimensionSizeToAddArray = table.create(offset, 1)

	local expandedTensor = self:expandNumberOfDimensions(dimensionSizeToAddArray)

	local targetDimension1 = dimensionArray[1] + offset
	local targetDimension2 = dimensionArray[2] + offset

	local expandedDimensionSizeArray = expandedTensor:getDimensionSizeArray()

	dimensionArray = {targetDimension1, targetDimension2}

	expandedDimensionSizeArray[targetDimension1], expandedDimensionSizeArray[targetDimension2] = expandedDimensionSizeArray[targetDimension2], expandedDimensionSizeArray[targetDimension1]

	local transposedTensor = createTensor(expandedDimensionSizeArray, true)

	if (table.find(dimensionArray, 1)) and (table.find(dimensionArray, 2)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							transposedTensor[a][b][c][d][e] = expandedTensor[b][a][c][d][e]

						end

					end

				end

			end

		end

	elseif (table.find(dimensionArray, 1)) and (table.find(dimensionArray, 3)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							transposedTensor[a][b][c][d][e] = expandedTensor[c][b][a][d][e]

						end

					end

				end

			end

		end

	elseif (table.find(dimensionArray, 2)) and (table.find(dimensionArray, 3)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							transposedTensor[a][b][c][d][e] = expandedTensor[a][c][b][d][e]

						end

					end

				end

			end

		end

	elseif (table.find(dimensionArray, 1)) and (table.find(dimensionArray, 4)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							transposedTensor[a][b][c][d][e] = expandedTensor[d][b][c][a][e]

						end

					end

				end

			end

		end

	elseif (table.find(dimensionArray, 1)) and (table.find(dimensionArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							transposedTensor[a][b][c][d][e] = expandedTensor[e][b][c][d][a]

						end

					end

				end

			end

		end

	elseif (table.find(dimensionArray, 2)) and (table.find(dimensionArray, 4)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							transposedTensor[a][b][c][d][e] = expandedTensor[a][d][c][b][e]

						end

					end

				end

			end

		end

	elseif (table.find(dimensionArray, 2)) and (table.find(dimensionArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							transposedTensor[a][b][c][d][e] = expandedTensor[a][e][c][d][b]

						end

					end

				end

			end

		end

	elseif (table.find(dimensionArray, 3)) and (table.find(dimensionArray, 4)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							transposedTensor[a][b][c][d][e] = expandedTensor[a][b][d][c][e]

						end

					end

				end

			end

		end

	elseif (table.find(dimensionArray, 3)) and (table.find(dimensionArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							transposedTensor[a][b][c][d][e] = expandedTensor[a][b][e][d][c]

						end

					end

				end

			end

		end

	elseif (table.find(dimensionArray, 4)) and (table.find(dimensionArray, 5)) then

		for a = 1, expandedDimensionSizeArray[1], 1 do

			for b = 1, expandedDimensionSizeArray[2], 1 do

				for c = 1, expandedDimensionSizeArray[3], 1 do

					for d = 1, expandedDimensionSizeArray[4], 1 do

						for e = 1, expandedDimensionSizeArray[5], 1 do

							transposedTensor[a][b][c][d][e] = expandedTensor[a][b][c][e][d]

						end

					end

				end

			end

		end

	else

		error("Invalid dimensions!")

	end
	
	transposedTensor = truncate(transposedTensor, offset)

	return AqwamTensorLibrary.new(transposedTensor)

end

function AqwamTensorLibrary:transpose(tensor, dimensionArray)

	local dimensionSizeArray = self:getDimensionSizeArray()

	if (#dimensionArray ~= 2) then error("Dimension array must contain 2 dimensions.") end

	local dimension1 = dimensionArray[1]

	local dimension2 = dimensionArray[2]

	local numberOfDimensions = #dimensionSizeArray

	if (dimension1 <= 0) then error("The first dimension must be greater than zero.") end

	if (dimension2 <= 0) then error("The second dimension must be greater than zero.") end

	if (dimension1 > numberOfDimensions) then error("The first dimension exceeds the tensor's number of dimensions") end

	if (dimension2 > numberOfDimensions) then error("The second dimension exceeds the tensor's number of dimensions") end

	if (dimension1 == dimension2) then error("The first dimension is equal to the second dimension.") end

	local transposedDimensionSizeArray = table.clone(dimensionSizeArray)

	transposedDimensionSizeArray[dimension1] = dimensionSizeArray[dimension2]

	transposedDimensionSizeArray[dimension2] = dimensionSizeArray[dimension1]

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	local transposedTensor = createTensor(transposedDimensionSizeArray, true)

	repeat

		local transposedDimensionIndexArray = table.clone(dimensionIndexArray)

		transposedDimensionIndexArray[dimension1] = dimensionIndexArray[dimension2]

		transposedDimensionIndexArray[dimension2] = dimensionIndexArray[dimension1]

		local value = self:getValue(dimensionIndexArray)

		setValue(transposedTensor, transposedDimensionSizeArray, value, transposedDimensionIndexArray)

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return AqwamTensorLibrary.new(transposedTensor)

end

local function containNoFalseBooleanInTensor(booleanTensor, dimensionSizeArray)

	if (#dimensionSizeArray > 1) then

		for i = 1, dimensionSizeArray[1], 1 do 

			local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

			if (not containNoFalseBooleanInTensor(booleanTensor[i], remainingDimensionSizeArray)) then return false end

		end

	else

		for i = 1, dimensionSizeArray[1], 1 do 

			if (not booleanTensor[i]) then return false end

		end

	end

	return true

end

function AqwamTensorLibrary:__eq(other)
	
	local tensorDimensionSizeArray = self:getDimensionSizeArray()

	local otherDimensionSizeArray = other:getDimensionSizeArray()

	if (not checkIfDimensionIndexArraysAreEqual(tensorDimensionSizeArray, otherDimensionSizeArray)) then return false end

	local booleanTensor = self:isEqualTo(other)

	return containNoFalseBooleanInTensor(booleanTensor, tensorDimensionSizeArray)

end

function AqwamTensorLibrary:isEqualTo(...)

	local functionToApply = function(a, b) return (a == b) end

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:isGreaterThan(...)

	local functionToApply = function(a, b) return (a > b) end

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:isGreaterOrEqualTo(...)

	local functionToApply = function(a, b) return (a >= b) end

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:isLessThan(...)

	local functionToApply = function(a, b) return (a < b) end

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:isLessOrEqualTo(...)

	local functionToApply = function(a, b) return (a <= b) end

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:tensorProduct(other)

	local resultTensor = tensorProduct(self, other)

	return AqwamTensorLibrary.new(resultTensor)

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

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a + b) end, self, other)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:add(...)

	local functionToApply = function(a, b) return (a + b) end

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:__sub(other)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a - b) end, self, other)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:subtract(...)

	local functionToApply = function(a, b) return (a - b) end

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:__mul(other)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a * b) end, self, other)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:multiply(...)

	local functionToApply = function(a, b) return (a * b) end

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:__div(other)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a / b) end, self, other)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:divide(...)

	local functionToApply = function(a, b) return (a / b) end

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:__unm()

	local resultTensor = applyFunctionOnMultipleTensors(function(a) return (-a) end, self)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:unaryMinus(...)

	local functionToApply = function(a) return (-a) end

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:logarithm(...)

	local functionToApply = math.log

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:exponent(...)

	local functionToApply = math.exp

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:power(...)

	local functionToApply = math.power

	local resultTensor

	if (self.tensor) then

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, self, ...)

	else

		resultTensor = applyFunctionOnMultipleTensors(functionToApply, ...)

	end

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:__pow(other)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a ^ b) end, self, other)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:__mod(other)

	local resultTensor = applyFunctionOnMultipleTensors(function(a, b) return (a % b) end, self, other)

	return AqwamTensorLibrary.new(resultTensor)

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

	local dimensionSizeArray = self:getDimensionSizeArray()

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

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	local extractedDimensionIndexArray = table.create(numberOfDimensions, 1)

	local extractedDimensionSizeArray = {}

	for i, targetDimensionIndex in ipairs(targetDimensionIndexArray) do extractedDimensionSizeArray[i] = targetDimensionIndex - originDimensionIndexArray[i] end

	for i, dimensionSize in ipairs(extractedDimensionSizeArray) do extractedDimensionSizeArray[i] = math.abs(dimensionSize) end

	local extractedTensor = createTensor(extractedDimensionSizeArray, true)

	repeat

		if checkIfDimensionIndexArrayIsWithinBounds(dimensionIndexArray, isDimensionIndexArrayDirectionSwappedArray, originDimensionIndexArray, targetDimensionIndexArray) then

			local copiedNewDimensionIndexArray = table.clone(extractedDimensionIndexArray)

			for i, boolean in ipairs(isDimensionIndexArrayDirectionSwappedArray) do

				if (boolean) then copiedNewDimensionIndexArray[i] = (extractedDimensionSizeArray[i] - copiedNewDimensionIndexArray[i]) + 1 end

			end

			local value = self:getValue(dimensionIndexArray)

			setValue(extractedTensor, extractedDimensionSizeArray, value, extractedDimensionIndexArray)

			extractedDimensionIndexArray = incrementDimensionIndexArray(extractedDimensionIndexArray, extractedDimensionSizeArray)

		end

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return AqwamTensorLibrary.new(extractedTensor)

end

local function dotProduct(tensor1, tensor2, tensor1DimensionSizeArray, tensor2DimensionSizeArray) -- Best one. Do not delete!

	local tensor1NumberOfDimensions = #tensor1DimensionSizeArray

	local tensor2NumberOfDimensions = #tensor2DimensionSizeArray

	local tensor = {}

	if (tensor1NumberOfDimensions == 1) and (tensor2NumberOfDimensions == 2) then

		for i = 1, #tensor1, 1 do -- Last dimension, so represents columns.

			tensor[i] = 0

			for j = 1, #tensor2[1], 1 do tensor[i] = (tensor1[i] * tensor2[i][j]) end -- Since tensor 1 column size matches with tensor 2 row size, we can use column index from tensor 1.

		end

	elseif (tensor1NumberOfDimensions == 2) and (tensor2NumberOfDimensions == 2) then

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

	elseif (tensor1NumberOfDimensions > 1) and (tensor2NumberOfDimensions > 2) then

		local remainingTensor1DimensionSizeArray = removeFirstValueFromArray(tensor1DimensionSizeArray)

		local remainingTensor2DimensionSizeArray = removeFirstValueFromArray(tensor2DimensionSizeArray)

		for i = 1, tensor1DimensionSizeArray[1] do tensor[i] = dotProduct(tensor1[i], tensor2[i], remainingTensor1DimensionSizeArray, remainingTensor2DimensionSizeArray) end

	elseif (tensor1NumberOfDimensions > 1) and (tensor2NumberOfDimensions == 2) then

		local remainingTensor1DimensionSizeArray = removeFirstValueFromArray(tensor1DimensionSizeArray)

		for i = 1, tensor1DimensionSizeArray[1] do tensor = dotProduct(tensor1[i], tensor2, remainingTensor1DimensionSizeArray, tensor2DimensionSizeArray) end

	elseif (tensor1NumberOfDimensions == 1) and (tensor2NumberOfDimensions > 2) then

		local remainingTensor2DimensionSizeArray = removeFirstValueFromArray(tensor2DimensionSizeArray)

		for i = 1, tensor2DimensionSizeArray[1] do tensor = dotProduct(tensor1, tensor2[i], tensor1DimensionSizeArray, remainingTensor2DimensionSizeArray) end

	elseif (tensor1NumberOfDimensions > 1) and (tensor2NumberOfDimensions == 1) then

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

	elseif (tensor1NumberOfDimensions == 0) or (tensor2NumberOfDimensions == 0) then

		tensor = tensor1:multiply(tensor2)

	else

		error("Unable to dot product.")

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

local function recursiveExpandedDotProduct(tensor1, tensor2, tensor1DimensionSizeArray, tensor2DimensionSizeArray) -- Since both have equal number of dimensions now, we only need to use only one dimension size array.

	local tensor1NumberOfDimensions = #tensor1DimensionSizeArray

	local tensor2NumberOfDimensions = #tensor2DimensionSizeArray

	local tensor

	if (tensor1NumberOfDimensions >= 3) and (tensor2NumberOfDimensions >= 3) and (tensor1DimensionSizeArray[1] == tensor2DimensionSizeArray[1]) then

		tensor = {}

		local remainingDimensionSizeArray1 = removeFirstValueFromArray(tensor1DimensionSizeArray)

		local remainingDimensionSizeArray2 = removeFirstValueFromArray(tensor2DimensionSizeArray)

		for i = 1, tensor1DimensionSizeArray[1], 1 do tensor[i] = recursiveExpandedDotProduct(tensor1[i], tensor2[i], remainingDimensionSizeArray1, remainingDimensionSizeArray2) end

	elseif (tensor1NumberOfDimensions == 2) and (tensor2NumberOfDimensions == 2) and (tensor1DimensionSizeArray[2] == tensor2DimensionSizeArray[1]) then -- No need an elseif statement where number of dimension is 1. This operation requires 2D tensors.

		tensor = tensor2DimensionalDotProduct(tensor1, tensor2)

	elseif (tensor1NumberOfDimensions == 0) or (tensor2NumberOfDimensions == 0) then

		tensor = AqwamTensorLibrary:multiply(tensor1, tensor2)

	elseif (tensor1NumberOfDimensions >= 2) and (tensor2NumberOfDimensions >= 2) and (tensor1DimensionSizeArray[1] ~= tensor2DimensionSizeArray[1]) then

		error("Unable to dot product. The starting dimension sizes of the first tensor does not equal to the starting dimension sizes of the second tensor.")

	else

		error("Unable to dot product.")

	end

	return tensor

end

local function expandedDotProduct(tensor1, tensor2)

	local dimensionSizeArray1 =  tensor1:getDimensionSizeArray()

	local dimensionSizeArray2 =  tensor2:getDimensionSizeArray()

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

		expandedTensor1 = tensor1:increaseNumberOfDimensions(dimensionSizeToAddArray)

	else

		expandedTensor1 = tensor1

	end

	if (numberOfDimensionsOffset2 ~= 0) then

		local dimensionSizeToAddArray = {}

		for i = 1, numberOfDimensionsOffset2, 1 do table.insert(dimensionSizeToAddArray, dimensionSizeArray1[i]) end

		expandedTensor2 = tensor2:increaseNumberOfDimensions(dimensionSizeToAddArray)

	else

		expandedTensor2 = tensor2

	end

	local expandedTensor1DimensionSizeArray = expandedTensor1:getDimensionSizeArray()

	local expandedTensor2DimensionSizeArray = expandedTensor2:getDimensionSizeArray()

	return recursiveExpandedDotProduct(expandedTensor1, expandedTensor2, expandedTensor1DimensionSizeArray, expandedTensor2DimensionSizeArray)

end

local function hardcodedDotProduct(tensor1, tensor2)

	local numberOfDimensions1 = tensor1:getNumberOfDimensions()

	local numberOfDimensions2 = tensor2:getNumberOfDimensions()

	local numberOfDimensionsOffset1 = 5 - numberOfDimensions1

	local numberOfDimensionsOffset2 = 5 - numberOfDimensions2

	local expandedTensor1 = tensor1:increaseNumberOfDimensions(table.create(numberOfDimensionsOffset1, 1))

	local expandedTensor2 = tensor2:increaseNumberOfDimensions(table.create(numberOfDimensionsOffset2, 1))

	local expandedNumberOfDimension1 = expandedTensor1:getDimensionSizeArray()

	local expandedNumberOfDimension2 = expandedTensor2:getDimensionSizeArray()

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

function AqwamTensorLibrary:dotProduct(...) -- Refer to this article. It was a fucking headache to do this. https://medium.com/@hunter-j-phillips/a-simple-introduction-to-tensors-c4a8321efffc

	local tensorArray = {...}

	if (self.tensor) then table.insert(tensorArray, 1, self) end

	local tensor = tensorArray[1]

	for i = 2, #tensorArray, 1 do tensor = expandedDotProduct(tensor, tensorArray[i]) end

	return AqwamTensorLibrary.new(tensor)

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

	local dimensionSize = #tensor

	local text = " "

	if (#dimensionSizeArray > 1) then

		local spacing = ""

		text = text .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSize, 1 do

			if (i > 1) then text = text .. spacing end

			text = text .. generateTensorString(tensor[i], remainingDimensionSizeArray, textSpacingArray, dimensionDepth + 1)

			if (i < dimensionSize) then text = text .. "\n" end



		end

		text = text .. " }"

	else

		text = text .. "{ "

		for i = 1, dimensionSize, 1 do

			local cellValue = tensor[i]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = textSpacingArray[i] - cellWidth

			text = text .. string.rep(" ", padding) .. cellText

			if (i < dimensionSize) then text = text .. " " end

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

local function generateTensorWithCommaString(tensor, dimensionSizeArray, textSpacingArray, dimensionDepth)

	local dimensionSize = #tensor

	local text = " "

	if (#dimensionSizeArray > 1) then

		local spacing = ""

		text = text .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSize, 1 do

			if (i > 1) then text = text .. spacing end

			text = text .. generateTensorWithCommaString(tensor[i], remainingDimensionSizeArray, textSpacingArray, dimensionDepth + 1)

			if (i < dimensionSize) then text = text .. "\n" end

		end

		text = text .. " }"

	else

		text = text .. "{ "

		for i = 1, dimensionSize, 1 do 

			local cellValue = tensor[i]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = textSpacingArray[i] - cellWidth

			text = text .. string.rep(" ", padding) .. cellText

			if (i < dimensionSize) then text = text .. ", " end

		end

		text = text .. " }"

	end

	return text

end

function AqwamTensorLibrary:generateTensorStringWithComma()

	local dimensionSizeArray = self:getDimensionSizeArray()

	local textSpacingArray = self:get2DTensorTextSpacing()

	return generateTensorWithCommaString(self, dimensionSizeArray, textSpacingArray, 1)

end

local function generatePortableTensorString(tensor, dimensionSizeArray, textSpacingArray, dimensionDepth)

	local dimensionSize = #tensor

	local text = " "

	if (#dimensionSizeArray > 1) then

		local spacing = ""

		text = text .. "{"

		for i = 1, dimensionDepth, 1 do spacing = spacing .. "  " end

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSize, 1 do

			if (i > 1) then text = text .. spacing end

			text = text .. generatePortableTensorString(tensor[i], remainingDimensionSizeArray, textSpacingArray, dimensionDepth + 1)

			if (i < dimensionSize) then text = text .. "\n" end

		end

		text = text .. " }"

		if (dimensionDepth > 1) then text = text .. "," end

	else

		text = text .. "{ "

		for i = 1, dimensionSize, 1 do 

			local cellValue = tensor[i]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = textSpacingArray[i] - cellWidth

			text = text .. string.rep(" ", padding) .. cellText

			if (i < dimensionSize) then text = text .. ", " end

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

	print("\n\n" .. self:generateTensorWithCommaString() .. "\n\n")

end

function AqwamTensorLibrary:printPortableTensor()

	print("\n\n" .. self:generatePortableTensorString() .. "\n\n")

end

local function getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

	local totalSize = 1

	for _, value in ipairs(dimensionSizeArray) do totalSize = value * totalSize end

	return totalSize

end

local function flattenAlongSpecifiedDimensions(dimensionSizeArray, startDimension, endDimension)

	local newDimensionSizeArray = {}

	local flattenedDimensionSize = 1

	for dimension, size in ipairs(dimensionSizeArray) do

		if (dimension >= startDimension) and (dimension <= endDimension) then flattenedDimensionSize = flattenedDimensionSize * size end

		if (dimension == endDimension) then table.insert(newDimensionSizeArray, flattenedDimensionSize) end

		if (dimension < startDimension) or (dimension > endDimension) then table.insert(newDimensionSizeArray, size) end

	end

	return newDimensionSizeArray

end

function AqwamTensorLibrary:flatten(dimensionArray)

	dimensionArray = dimensionArray or {}

	local dimensionSizeArray = self:getDimensionSizeArray()

	local numberOfDimensions = #dimensionSizeArray

	local startDimension = dimensionArray[1] or 1

	local endDimension = dimensionArray[2] or numberOfDimensions

	if (endDimension == math.huge) then endDimension = numberOfDimensions end

	local newDimensionSizeArray = flattenAlongSpecifiedDimensions(dimensionSizeArray, startDimension, endDimension)

	return self:reshape(newDimensionSizeArray)

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

local function reshape(tensor, dimensionSizeArray, targetTensor, targetDimensionSizeArray, currentTargetDimensionIndexArray)

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do 

			currentTargetDimensionIndexArray = reshape(tensor[i], remainingDimensionSizeArray, targetTensor, targetDimensionSizeArray, currentTargetDimensionIndexArray) 

		end

	else

		for i = 1, dimensionSizeArray[1], 1 do 

			targetTensor:setValue(tensor[i], currentTargetDimensionIndexArray)

			currentTargetDimensionIndexArray = incrementDimensionIndexArray(currentTargetDimensionIndexArray, targetDimensionSizeArray)

		end

	end

	return currentTargetDimensionIndexArray

end

function AqwamTensorLibrary:inefficientReshape(dimensionSizeArray) -- This one requires higher space complexity due to storing the target dimension index array for each of the values. It is also less efficient because it needs to use recursion to get and set values from and to the target tensor.

	local tensorDimensionSizeArray = self:getDimensionSizeArray()

	local totalNumberOfValue = getTotalSizeFromDimensionSizeArray(tensorDimensionSizeArray)

	local totalNumberOfValuesRequired = getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

	if (totalNumberOfValue ~= totalNumberOfValuesRequired) then error("The number of values of the tensor does not equal to total number of values of the reshaped tensor.") end

	local resultTensor

	if (#tensorDimensionSizeArray ~= 1) then

		resultTensor = AqwamTensorLibrary:createTensor(dimensionSizeArray, true)

		local currentTargetDimensionIndexArray = table.create(#dimensionSizeArray, 1)

		reshape(self, tensorDimensionSizeArray, resultTensor, dimensionSizeArray, currentTargetDimensionIndexArray)

	else

		resultTensor = reshapeFromFlattenedTensor(self, dimensionSizeArray, 1)

	end

	return AqwamTensorLibrary.new(resultTensor)

end

local function flattenTensor(tensor, dimensionSizeArray, targetTensor)

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do flattenTensor(tensor[i], remainingDimensionSizeArray, targetTensor) end

	else

		for _, value in ipairs(tensor) do table.insert(targetTensor, value) end

	end

end

function AqwamTensorLibrary:reshape(dimensionSizeArray) -- This one requires lower space complexity as it only need to flatten the tensor. Then only need a single target dimension index array that will be used by all values from the original tebsor.

	local tensorDimensionSizeArray = self:getDimensionSizeArray()

	local totalSize = getTotalSizeFromDimensionSizeArray(tensorDimensionSizeArray)

	local totalSizeRequired = getTotalSizeFromDimensionSizeArray(dimensionSizeArray)

	if (totalSize ~= totalSizeRequired) then error("The total size of the tensor does not equal to the total size of the reshaped tensor.") end

	local flattenedTensor

	if (#tensorDimensionSizeArray ~= 1) then

		flattenedTensor = {}

		flattenTensor(self, tensorDimensionSizeArray, flattenedTensor)

	else

		flattenedTensor = self

	end

	local resultTensor = reshapeFromFlattenedTensor(flattenedTensor, dimensionSizeArray, 1)

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:squeeze(dimension)

	if (type(dimension) ~= "number") then error("The dimension must be a number.") end

	local dimensionSizeArray = self:getDimensionSizeArray()

	local numberOfDimensions = #dimensionSizeArray

	throwErrorIfDimensionIsOutOfBounds(dimension, 1, numberOfDimensions)

	if (dimensionSizeArray[dimension] ~= 1) then error("The dimension size at dimension " .. dimension .. " is not equal to 1.") end

	local resultDimensionSizeArray = table.clone(dimensionSizeArray)

	table.remove(resultDimensionSizeArray, dimension)

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(#dimensionSizeArray, 1)

	local resultTensor = AqwamTensorLibrary:createTensor(resultDimensionSizeArray, true)

	repeat

		local resultDimensionIndexArray = table.clone(dimensionIndexArray)

		table.remove(resultDimensionIndexArray, dimension)

		local value = self:getValue(dimensionIndexArray)

		setValue(resultTensor, resultDimensionSizeArray, value, resultDimensionIndexArray)

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return AqwamTensorLibrary.new(resultTensor)

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

function AqwamTensorLibrary:findMaximumValue()

	local dimensionSizeArray = self:getDimensionSizeArray()

	local numberOfDimensions = #dimensionSizeArray

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	local highestValue = -math.huge

	repeat

		local value = self:getValue(dimensionIndexArray)

		highestValue = math.max(highestValue, value)

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return highestValue

end

function AqwamTensorLibrary:findMinimumValue()

	local dimensionSizeArray = self:getDimensionSizeArray()

	local numberOfDimensions = #dimensionSizeArray

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	local lowestValue = math.huge

	repeat

		local value = self:getValue(dimensionIndexArray)

		lowestValue = math.min(lowestValue, value)

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return lowestValue

end

function AqwamTensorLibrary:findMaximumValueDimensionIndexArray()

	local dimensionSizeArray = self:getDimensionSizeArray()

	local numberOfDimensions = #dimensionSizeArray

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	local highestValueDimensionIndexArray

	local highestValue = -math.huge

	repeat

		local value = self:getValue(dimensionIndexArray)

		if (value > highestValue) then

			highestValueDimensionIndexArray = table.clone(dimensionIndexArray)

			highestValue = value

		end

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return highestValueDimensionIndexArray, highestValue

end

function AqwamTensorLibrary:findMinimumValueDimensionIndexArray()

	local dimensionSizeArray = self:getDimensionSizeArray()

	local numberOfDimensions = #dimensionSizeArray

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	local lowestValueDimensionIndexArray

	local lowestValue = math.huge

	repeat

		local value = self:getValue(dimensionIndexArray)

		if (value < lowestValue) then

			lowestValueDimensionIndexArray = table.clone(dimensionIndexArray)

			lowestValue = value

		end

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return lowestValueDimensionIndexArray, lowestValue

end

function AqwamTensorLibrary:destroy()

	self.tensor = nil

	setmetatable(self, nil)

end

function AqwamTensorLibrary:isSameTensor(...)
	
	local tensorArray = {...}
	
	if (self.tensor) then table.insert(tensorArray, 1, self) end
	
	for i = 1, (#tensorArray - 1) do
		
		local tensor1 = tensorArray[i]
		
		local tensor2 = tensorArray[i + 1]
		
		local tensor1DimensionSizeArray = tensor1:getDimensionSizeArray()

		local tensor2DimensionSizeArray = tensor2:getDimensionSizeArray()
		
		if (not checkIfDimensionIndexArraysAreEqual(tensor1DimensionSizeArray, tensor2DimensionSizeArray)) then return false end
		
		local booleanTensor = tensor1:isEqualTo(tensor2)
		
		if (not containNoFalseBooleanInTensor(booleanTensor, tensor1DimensionSizeArray)) then return false end
		
	end

	return true 

end

local function applyFunction(functionToApply, dimensionSizeArray, ...)

	local tensorArray = {...}

	local resultTensor = {}

	if (#dimensionSizeArray >= 2) then

		local remainingDimensionSizeArray = removeFirstValueFromArray(dimensionSizeArray)

		for i = 1, dimensionSizeArray[1], 1 do 

			local subTensorArray = {}

			for _, tensor in ipairs(tensorArray) do table.insert(subTensorArray, tensor[i]) end

			resultTensor[i] = applyFunction(functionToApply, remainingDimensionSizeArray, table.unpack(subTensorArray)) 

		end

	else

		for i = 1, dimensionSizeArray[1], 1 do 

			local subTensorArray = {}

			for _, tensor in ipairs(tensorArray) do table.insert(subTensorArray, tensor[i]) end

			resultTensor[i] = functionToApply(table.unpack(subTensorArray)) 

		end

	end

	return resultTensor

end

function AqwamTensorLibrary:applyFunction(functionToApply, ...)

	local tensorArray = {...}

	if (self.tensor) then table.insert(tensorArray, 1, self) end

	local numberOfTensors = #tensorArray

	if (numberOfTensors >= 2) then

		for i = 1, (numberOfTensors - 1), 1 do tensorArray[i], tensorArray[i + 1] = broadcast(tensorArray[i], tensorArray[i + 1], false) end

	end

	local dimensionSizeArray = tensorArray[1]:getDimensionSizeArray()

	local resultTensor = applyFunction(functionToApply, dimensionSizeArray, table.unpack(tensorArray))

	return AqwamTensorLibrary.new(resultTensor)

end

function AqwamTensorLibrary:permute(dimensionArray)

	local dimensionSizeArray = self:getDimensionSizeArray()

	local numberOfDimensions = #dimensionSizeArray

	if (numberOfDimensions ~= #dimensionArray) then error("The number of dimensions does not match.") end

	local collectedDimensionArray = {}

	for i, dimension in ipairs(dimensionArray) do

		if (dimension > numberOfDimensions) then error("Value of " .. dimension .. " in the target dimension array exceeds the number of dimensions.") end

		if (table.find(collectedDimensionArray, dimension)) then error("Value of " .. dimension .. " in the target dimension array has been added more than once.") end

		table.insert(collectedDimensionArray, dimension)

	end

	local dimensionIndexArray = table.create(numberOfDimensions, 1)

	local dimensionIndexArrayToEndLoop = table.create(numberOfDimensions, 1)

	local permutedDimensionIndexArray = {}

	local permutedDimensionSizeArray = {}

	for i, dimension in ipairs(dimensionArray) do permutedDimensionSizeArray[i] = dimensionSizeArray[dimension] end

	local permutedTensor = createTensor(permutedDimensionSizeArray, true)

	repeat

		for i, dimension in ipairs(dimensionArray) do permutedDimensionIndexArray[i] = dimensionIndexArray[dimension] end

		local value = self:getValue(dimensionIndexArray)

		setValue(permutedTensor, permutedDimensionSizeArray, value, permutedDimensionIndexArray)

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

		dimensionIndexArray = incrementDimensionIndexArray(dimensionIndexArray, dimensionSizeArray)

	until checkIfDimensionIndexArraysAreEqual(dimensionIndexArray, dimensionIndexArrayToEndLoop)

	return AqwamTensorLibrary.new(permutedTensor)

end

return AqwamTensorLibrary
